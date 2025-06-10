import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('data/songs.csv')

# Combine features
df['combined'] = df['artist'] + ' ' + df['genre']

# Vectorize text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommend function
def recommend(song_title):
    if song_title not in df['title'].values:
        print(f"'{song_title}' not found in the database.")
        return
    idx = df[df['title'] == song_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    print(f"Recommendations for '{song_title}':")
    for i in sim_scores:
        print(f"- {df.iloc[i[0]]['title']}")

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a song title: ")
    recommend(user_input)
