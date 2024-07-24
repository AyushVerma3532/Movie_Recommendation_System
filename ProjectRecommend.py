import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def binary_string_to_vector(binary_string):
    res=[]
    while len(res)<19-len(binary_string):
        res.append(0.0)
    for bit in binary_string:
        res.append(float(bit))
    
    return torch.tensor(res)
    
def cosine_similarity(A, B):
    dot_product = torch.dot(A, B)
    norm_a = torch.norm(A)
    norm_b = torch.norm(B)
    return dot_product / (norm_a * norm_b)


#Optimizing the user_factors to make sure it gives as close ratings to the original provided by the user
#this will help us know how would a user rate other unseen movies
class NewUserRatings(nn.Module):
    def __init__(self, num_factors):
        super().__init__()
        self.user_factors = nn.Embedding(1, num_factors)
        self.user_factors.weight.data.uniform_(0,0.1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, item_factors):
        user_factors = self.user_factors.weight
        dot_product= torch.matmul(user_factors,  item_factors.T)
        return dot_product

    def ratingLoss(y_true, y_pred):
        squared_diff = (y_true - y_pred)**2
        return squared_diff.sum()
num_factors=20
model_1=NewUserRatings(num_factors)

item_factors = torch.load('C:\\Users\\NITRO\\Downloads\\item_factors.pth') #Previously trained item_factors
movie_titles = pd.read_csv('C:\\Users\\NITRO\\Downloads\\movies3.csv')  # Columns: ['movie_id', 'title']
movie_titles['genre_vector']=movie_titles['genre_vector'].astype('int64')
Genres=movie_titles.columns.to_list()[4:23]
movie_titles['genre_vector']=movie_titles['genre_vector'].astype(str)
print(type(movie_titles['genre_vector'][0]))
item_factors_np = item_factors.numpy()

st.title('Movie Recommender System')
st.write('Please select some movies you watched and rate them. The more movies you rate, the better will be the recommendations')

# Initialize session state
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = []

# Search bar for movies
search_query = st.text_input('Search for movies:')

if search_query:
    matching_movies = movie_titles[
    movie_titles['title'].str.contains(search_query, case=False, na=False) |
    (movie_titles['title'].str.lower() == search_query.lower())
]
    if not matching_movies.empty:
        # Create a dropdown list for movie selection
        selected_movie = st.selectbox('Select a movie to rate:', options=matching_movies['title'].tolist())

        # Display selected movie and rating slider
        if selected_movie:
            selected_movie_id = matching_movies[matching_movies['title'] == selected_movie]['movieId1'].values[0]
            rating = st.slider(f"Rate '{selected_movie}'", 1, 5, 3, key=selected_movie_id)

            if st.button('Add Rating'):
                # Store the rated movie and rating
                st.session_state.rated_movies.append({'movieId1': selected_movie_id, 'title': selected_movie, 'rating': rating})
                st.success(f"Added rating for '{selected_movie}'")

            # Display the list of rated movies
            if st.session_state.rated_movies:
                st.write("Rated Movies:")
                rated_movies_df = pd.DataFrame(st.session_state.rated_movies)
                st.write(rated_movies_df.drop(columns='movieId1'))

            # Collect user ratings
            rated_movie_ids = [movie['movieId1'] for movie in st.session_state.rated_movies]
            ratings = [movie['rating'] for movie in st.session_state.rated_movies]

            if st.button('Get Recommendations'):
                if len(rated_movie_ids) > 0:
                    rated_movie_factors=[]

                    #visited vector in which the movies that are rated will be marked 1 and that would act as a flag to avoid repetition in the recommended movies
                    visited_movies=torch.zeros(len(item_factors))
                    averaged_movie_vector=torch.zeros(19)
                    for i in range(len(rated_movie_ids)):
                        rated_movie_factors.append(np.array(item_factors[rated_movie_ids[i]-1]))
                        visited_movies[rated_movie_ids[i]-1]=1 
                        averaged_movie_vector+=(ratings[i]-2.5)*binary_string_to_vector(np.array(movie_titles[movie_titles['movieId1']==rated_movie_ids[i]]['genre_vector'])[0])
                    
                    rated_movie_factors=torch.tensor(np.array(rated_movie_factors))
                    ratings_new=model_1(rated_movie_factors)
                    epochs=1000
                    lr = 0.01
                    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
                    loss1=0

                    for epoch in range(epochs):
                        model_1.train()
                        prediction = model_1(rated_movie_factors)
                        loss = NewUserRatings.ratingLoss(torch.tensor(ratings), prediction)
                        loss1=loss
                        optimizer_1.zero_grad()
                        loss.backward()
                        optimizer_1.step()
                    #st.write(loss)
                    prediction = model_1(item_factors)
                    prediction1=1+4*torch.sigmoid(prediction-3)# considering the mean rating to be 3, the predictions are adjusted around 3 and maintained withing 1 to 5
                    recommendations=[]
                    tags=[]
                    prediction1=prediction1.squeeze().detach().numpy()
                    sorted_indices = np.argsort(prediction1)[::-1]

                    i=0
                    c=0
                    while i<len(sorted_indices):
                        if(visited_movies[sorted_indices[i]]!=1):
                            movie=np.array(movie_titles[movie_titles['movieId1']==sorted_indices[i]+1])
                            if len(movie)>0:
                                similarity=cosine_similarity(averaged_movie_vector, binary_string_to_vector(np.array(movie_titles[movie_titles['movieId1']==sorted_indices[i]+1]['genre_vector'])[0]))
                                if similarity>0.5:
                                    recommendations.append(movie_titles[movie_titles['movieId1'] == sorted_indices[i] + 1]['title'].values[0])
                                    tags.append(movie_titles[movie_titles['movieId1'] == sorted_indices[i] + 1]['genres'].values[0])
                                    c+=1
                        i+=1
                        if(c==20):
                            break
                    
                    st.write("Top 20 Recommendations:")
                    for title, tag in zip(recommendations, tags):
                        st.write(f"{title}, {tag}")

                else:
                    st.write("Please rate at least one movie to get recommendations.")
        else:
            st.write("Please select a movie to rate.")
    else:
        st.write("No movies found. Please adjust your search query.")
