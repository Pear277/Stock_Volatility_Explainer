import requests

class VolatilityExplainer:
    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_explanation(self, volatility_level, education_mode, trend_summary, investment_intent, indicator_summary,investment_horizon,vol_window):
        intent_context = f"The user is considering {investment_intent.lower()} this stock.\n"
        volatility_context = f"The current rolling volatility is {volatility_level:.2f}% over a {vol_window}-day window.\n"
        trend_context = f"Here is the trend summary:\n{trend_summary}\n"
        indicator_context = f"Here are the current indicators:\n{indicator_summary}\n"

        if investment_horizon == "Both":
            horizon_context = (
                "Provide two clearly labeled sections:\n"
                "**Short-Term Recommendation:**\n"
                "**Long-Term Recommendation:**\n"
                "Each should include one clear action (Buy, Hold, or Sell), justification based on indicators, and what signals to monitor going forward.\n"
                "Please include two clearly labeled lines:\n"
                "Short-Term Action: [Buy/Hold/Sell]\n"
                "Long-Term Action: [Buy/Hold/Sell]\n"
                "These should appear after your recommendations and justifications. Keep them on separate lines for easy parsing." 
            )
        elif investment_horizon == "Short-Term":
            horizon_context = (
                "Provide one clearly labeled section:\n"
                "**Short-Term Recommendation:**\n"
                "Include one clear action (Buy, Hold, or Sell), explain why, and what to watch for.\n"
                "Please include one clearly labeled line:\n"
                "Short-Term Action: [Buy/Hold/Sell]\n"
                "This should appear after your recommendation and justification."
            )
        elif investment_horizon == "Long-Term":
            horizon_context = (
                "Provide one clearly labeled section:\n"
                "**Long-Term Recommendation:**\n"
                "Include one clear action (Buy, Hold, or Sell), explain why, and what to watch for.\n"
                "Please include one clearly labeled line:\n"
                "Long-Term Action: [Buy/Hold/Sell]\n"
                "This should appear after your recommendation and justification."
            )
 

        if education_mode == "Beginner":
            prompt = intent_context + volatility_context + trend_context + indicator_context + horizon_context + (
                "\nExplain what the volatility score means in simple terms. "
                "Then explain the trend summary in plain language. "
                "Use analogies or examples if helpful. "
                "Make the recommendation easy to understand for a retail investor."
            )
        else:
            prompt = intent_context + volatility_context + trend_context + indicator_context + horizon_context + (
                "\nProvide a technical explanation of the volatility score and trend summary. "
                "Justify the recommendation using volatility, trend, and momentum indicators."
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

