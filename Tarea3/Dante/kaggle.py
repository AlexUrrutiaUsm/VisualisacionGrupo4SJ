import kagglehub

# Download latest version
path = kagglehub.dataset_download("prasertk/netflix-subscription-price-in-different-countries")

print("Path to dataset files:", path)