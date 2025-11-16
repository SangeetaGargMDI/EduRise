"""
generate_reports.py
Generative AI interface for EduRise.
Takes prediction + explanation and formats a prompt for an LLM.
"""

from typing import List, Tuple
from openai import OpenAI


# ============================================================
#  SET API KEY DIRECTLY (Your working key)
#  ⚠️ You MUST regenerate this key after testing.
# ============================================================
OPENAI_API_KEY = "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"

# Initialize client safely
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print("Failed to initialize OpenAI client:", e)
    client = None


# ============================================================
#  BUILD PROMPT
# ============================================================
def build_prompt(
    school_name: str,
    location: str,
    risk_bucket: str,
    drivers: List[Tuple[str, float]],
) -> str:

    drivers_sorted = sorted(drivers, key=lambda x: x[1], reverse=True)
    top = drivers_sorted[:5]
    drivers_str = ", ".join(f"{name} (weight {weight:.3f})" for name, weight in top)

    prompt = f"""
You are an education policy expert working with government schools in India.

School: {school_name}
Location: {location}
Predicted enrolment status: {risk_bucket}

Top contributing factors for this prediction: {drivers_str}.

You must provide three outputs:

[Action Plan] (150–200 words)
Concrete, low-cost interventions for the Headmaster and BEO within the next 6–12 months.

[Parent Message Hindi]
2–3 short, simple Hindi sentences for WhatsApp encouraging enrolment/retention.

[Policy Note]
3–4 bullet points for DEO summarizing structural issues and recommended medium-term actions.

Respond EXACTLY in this format:
[Action Plan]
...

[Parent Message Hindi]
...

[Policy Note]
- ...
- ...
- ...
"""
    return prompt


# ============================================================
#  CALL LLM
# ============================================================
def call_llm(prompt: str) -> str:

    if client is None:
        return "ERROR: OpenAI client not initialized."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert education policy assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"LLM ERROR: {e}"


# ============================================================
#  STANDALONE TEST RUN
# ============================================================
if __name__ == "__main__":

    test_prompt = build_prompt(
        school_name="GPS Ward 13",
        location="Jaipur, Rajasthan",
        risk_bucket="decline",
        drivers=[
            ("Pupil-teacher ratio", 0.23),
            ("Infrastructure index", 0.19),
            ("Attendance rate", 0.15),
            ("Community support", 0.10),
        ],
    )

    print("=== Prompt Preview ===")
    print(test_prompt)

    print("\n=== LLM Output ===")
    print(call_llm(test_prompt))
