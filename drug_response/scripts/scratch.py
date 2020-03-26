import pickle

shap_values = pickle.load(open("/home/scai/SangerDrug/work_dirs/shap/deep_shap.pkl", "rb"))

index = pickle.load(open("/home/scai/SangerDrug/work_dirs/shap/gradient_indexes.pkl", "rb"))
index = index.detach().cpu().numpy()
pickle.dump(index, open("/home/scai/SangerDrug/work_dirs/shap/gradient_indexes.pkl", "wb"))
shap_values = [x.detach().cpu().numpy() for x in shap_values]