from src.models.model_loader import ModelLoader

loader = ModelLoader()

loader.load_models()

models = loader.get_models()

print()

print("=" * 60)

print("Loaded Objects")

print("=" * 60)

for name in models:

    print(f"{name:20} : {type(models[name])}")