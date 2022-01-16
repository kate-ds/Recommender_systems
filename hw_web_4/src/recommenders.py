import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.pop_item = 0
        self.pop_list = 0

    @staticmethod
    def prepare_matrix(data):

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    def prepare_popularity_matrix(self,  n=5):
        """топ-N покупок пользователя"""

        popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]

        self.pop_item = popularity.head(1).item_id.values
        self.pop_list = popularity.head(n).item_id.to_list()

        popularity = popularity.groupby('user_id').head(n)
        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)

        return popularity


    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popularity = self.prepare_popularity_matrix(n=N)
        popular_user_items = popularity.item_id[popularity['user_id'] == user]

        res = []
        for item in popular_user_items:
            recs = self.model.similar_items(self.itemid_to_id[item], N=2)
            top_rec = recs[1][0]
            res.append(self.id_to_itemid[top_rec])

        # обработка пользователей, у которых меньше 5 покупок (оставшиеся товары рекомендуем из популярных)
        i = 0
        while len(res) != N:
            res.append(self.pop_list[i])
            i += 1

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        popularity = self.prepare_popularity_matrix(1)
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)

        res = []
        for i in range(1, N+1):
            similar_user = similar_users[i][0]
            popular_item = popularity[popularity['user_id'] == similar_user]['item_id'].values

            if len(popular_item) != 0:
                res.append(int(popular_item))
            else:
                try:
                    res.append(self.pop_item[0])
                except KeyError:
                    print(f'Error user {user}')

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
