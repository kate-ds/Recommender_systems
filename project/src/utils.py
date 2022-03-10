"""
Pre- and postfilter items
"""

import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None):
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    data = data[data['week_no'] > data['week_no'].max() - 52]
    
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999


    return data

def make_unique_recommendations(recommendations, N=100):
    
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]
    
    n_rec = len(unique_recommendations)
    if n_rec < N:
        final_recommendations = unique_recommendations.extend(unique_recommendations[:N - n_rec])
        #вствить топ рекомендаций
    else:
        final_recommendations = unique_recommendations[:N]
    
    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations

def postfilter_items(recommendations, item_features, N=100):

    # Уникальность
    
    # В ЭТОМ ПОСТФИЛЬТРЕ ОСОБОЙ НЕОБХОДИМОСТИ НЕТ, Т К УЖЕ БЫЛА ОБРАБОТКА, ЧТОБЫ РЕКОМЕНДАЦИИ ДАВАЛИ ТОЛЬКО УНИКАЛЬНЫЕ ЗНАЧЕНИЯ

    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    
    # Сделаем так, чтобы каждый товар был из разной категории
    categories_used = []
    final_recommendations = []
    
    DEPARTMENT_NAME = 'department'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, DEPARTMENT_NAME].values[0]
        
        if category not in categories_used:
            final_recommendations.append(item)
            
        unique_recommendations.remove(item)
        categories_used.append(category)
    
    n_rec = len(final_recommendations)
    
    if n_rec < N:
        final_recommendations.extend(unique_recommendations[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]
        

#     n_rec = len(unique_recommendations)
#     if n_rec < N:
#         final_recommendations = unique_recommendations.extend(unique_recommendations[:N - n_rec])
#         #вствить топ рекомендаций
#     else:
#         final_recommendations = unique_recommendations[:N]
    
    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations

