from llm_explainer import VolatilityExplainer

explainer = VolatilityExplainer(model_name="qwen2.5:7b")
response = explainer.generate_explanation(3.2)
print("Raw response:", response)
