import requests

class VolatilityExplainer:
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_explanation(self, volatility_level, education_mode, trend_summary,investment_intent):
        if education_mode == "Beginner":
            intent_context = f"The user is considering {investment_intent.lower()} this stock. "

            prompt = intent_context + (
                f"Explain in simple terms what a {volatility_level:.2f}% rolling volatility means for a retail investor. "
                f"Use analogies or everyday examples. "
                f"Then, based on the following indicators:\n{trend_summary}\n"
                f"Suggest whether the investor should Buy, Hold, or Sell, and explain why."
            )
        else:
            intent_context = f"The user is considering {investment_intent.lower()} this stock. "
            prompt = intent_context + (
                f"Provide a technical explanation of {volatility_level:.2f}% rolling volatility for an experienced investor. "
                f"Include risk classification, expected price movement range, and implications for trading strategy. "
                f"Then, based on the following indicators:\n{trend_summary}\n"
                f"Suggest whether the investor should Buy, Hold, or Sell, and justify your recommendation."
            )

        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            })

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"⚠️ API error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"⚠️ Exception occurred: {str(e)}"

