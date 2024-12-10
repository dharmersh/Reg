from astrapy import DataAPIClient

# Initialize the client
client = DataAPIClient("AstraCS:ahdUepZuDDSSvAnreccKyukG:6f634e3ace512774e3943ea5aee42ef6eaa817ce54e1c3d34563a65d8b1bb85b")
db = client.get_database_by_api_endpoint(
  "https://7127dc78-d109-4724-b26b-0b658ab52dc8-us-east1.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")

# Insert documents with embeddings into the collection.
documents = [
    {
        "text": "Chat bot integrated sneakers that talk to you",
        "$vector": [0.1, 0.15, 0.3, 0.12, 0.05],
    },
    {
        "text": "An AI quilt to help you sleep forever",
        "$vector": [0.45, 0.09, 0.01, 0.2, 0.11],
    },
    {
        "text": "A deep learning display that controls your mood",
        "$vector": [0.1, 0.05, 0.08, 0.3, 0.6],
    },
]
insertion_result = db.create_collection(documents)
print(f"* Inserted {len(insertion_result.inserted_ids)} items.\n")