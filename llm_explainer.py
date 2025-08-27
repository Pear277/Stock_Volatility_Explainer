import requests

class VolatilityExplainer:
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_explanation(self, volatility_level):
        prompt = (
            f"Explain in plain English what a {volatility_level:.2f}% rolling volatility means "
            f"for a retail investor. Include whether this level is considered stable, moderate, or risky, "
            f"and suggest what kind of investor might be comfortable with it."
        )

        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            })

            print("Status Code:", response.status_code)
            print("Raw JSON:", response.text)

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"⚠️ API error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"⚠️ Exception occurred: {str(e)}"
