
"""
Generative AI interface for EduRise.

Takes prediction + explanation and formats a prompt for an LLM.
"""

from typing import List, Tuple

def build_prompt(
    school_name: str,
    location: str,
    risk_bucket: str,
    drivers: List[Tuple[str, float]],
) -> str:
    """Build a structured prompt for an LLM based on school context."""
    drivers_sorted = sorted(drivers, key=lambda x: x[1], reverse=True)
    top = drivers_sorted[:5]
    drivers_str = ", ".join(f"{name} (weight {weight:.3f})" for name, weight in top)
    prompt = f"""You are an education policy expert working with government schools in India.
School: {school_name}
Location: {location}
Predicted enrolment status: {risk_bucket}

Top contributing factors for this prediction: {drivers_str}.

1) In 150–200 words, suggest concrete, low-cost interventions that the headmaster and Block Education Officer can implement in the next 6–12 months to improve enrolment.
2) Draft a 2–3 sentence message in simple Hindi that can be sent to parents on WhatsApp encouraging them to enroll or retain their children in this school.
3) Provide a short policy note (3–4 bullet points) for the District Education Officer summarizing the key structural issues and recommended medium-term actions.

Respond in the following structure:
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

def call_llm(prompt: str) -> str:
    """Placeholder for actual LLM API call.

    Replace with code that calls OpenAI, Azure OpenAI, or a local model.
    """
    return "LLM_CALL_NOT_IMPLEMENTED. Prompt begins:\n" + prompt[:500]

if __name__ == "__main__":
    demo_prompt = build_prompt(
        school_name="Govt Primary School Ward 13",
        location="Jaipur, Rajasthan",
        risk_bucket="decline",
        drivers=[("Pupil-teacher ratio", 0.23), ("Infrastructure index", 0.19)],
    )
    print(call_llm(demo_prompt))
