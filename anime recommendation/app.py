import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained models
pca_model = joblib.load('pca_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# Load anime and user rating data
anime = pd.read_csv('anime.csv')
user = pd.read_csv('rating.csv')

# Perform user-based filtering and clustering
MRPU = user.groupby(['user_id']).mean().reset_index()
MRPU['mean_rating'] = MRPU['rating']
MRPU.drop(['anime_id','rating'],axis=1, inplace=True)
user = pd.merge(user, MRPU, on=['user_id', 'user_id'])
user = user.drop(user[user.rating < user.mean_rating].index)

mergedata = pd.merge(anime, user, on=['anime_id', 'anime_id'])
mergedata = mergedata[mergedata.user_id <= 20000]

user_anime = pd.crosstab(mergedata['user_id'], mergedata['name'])

# Apply PCA using the pre-trained model
pca_samples = pca_model.transform(user_anime)
tocluster = pd.DataFrame(pca_samples)


optimal_clusters = 4
clusterer = kmeans_model.fit(tocluster)  # Use the pre-trained KMeans model
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)

# Add cluster information to user_anime
user_anime['cluster'] = c_preds

st.title('Anime Recommendation App')

# User selection
selected_user = st.selectbox('Виберіть айді користувача:', user_anime.index)

watched_anime = user_anime.loc[selected_user]
st.write(f"Аніме переглянуті користувачем {selected_user}:")
st.dataframe(watched_anime[watched_anime > 0].reset_index().rename(columns={'name': 'Anime Name', selected_user: 'rating'}))

# Recommendations based on user's cluster
selected_cluster = int(user_anime[user_anime.index == selected_user]['cluster'])
watched_anime_names = watched_anime[watched_anime > 0].index.tolist()

recommended_anime = user_anime[(user_anime['cluster'] == selected_cluster) & (~user_anime.index.isin(watched_anime_names))]

recommended_anime = recommended_anime.mean().sort_values(ascending=False).head(10)

st.write(f"Топ-10 рекомендацій для користувача {selected_user} (Кластер {selected_cluster}):")
st.dataframe(recommended_anime.reset_index().rename(columns={'name': 'Anime Name', 0: 'Average Rating'}))

fig = plt.figure(figsize=(10, 8))
plt.scatter(tocluster[1], tocluster[0], c=c_preds)
for ci, c in enumerate(centers):
    plt.plot(c[1], c[0], 'o', markersize=8, color='black', alpha=1)

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Data points in 2D PCA axis', fontsize=20)
st.pyplot(fig)






