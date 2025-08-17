# app.py
# Streamlit + browser-use job application copilot
# Setup:
#   python -m venv .venv && source .venv/bin/activate
#   pip install streamlit browser-use playwright pydantic
#   python -m playwright install
# Run:
#   streamlit run app.py

import asyncio
import json
import os
import re
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel, Field, HttpUrl, ValidationError

# ---- browser-use imports (stable across versions) ----
from browser_use import Agent
from browser_use.llm.openai import OpenAIChat
from browser_use.browser.context import BrowserContextConfig

# Controller import path moved across versions; try both.
try:
    from browser_use.controller.service import Controller
except Exception:  # pragma: no cover
    from browser_use.controller.controller import Controller  # older releases

URL_REGEX = r"(https?://[^\s]+)"


# ---------------- Models ----------------
class Profile(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str = ""
    portfolio_url: Optional[HttpUrl] = None
    linkedin_url: Optional[HttpUrl] = None
    github_url: Optional[HttpUrl] = None
    work_authorization: str = "Authorized to work in the U.S."
    desired_salary: str = ""
    notice_period: str = ""
    resume_url: str = Field(..., description="Public URL to resume (PDF preferred)")


class JobBatch(BaseModel):
    job_urls: List[HttpUrl]


def extract_urls(text: str) -> List[str]:
    return re.findall(URL_REGEX, text or "")


def build_goal(profile: Profile, job_urls: List[str]) -> str:
    links = [
        f"- Portfolio: {profile.portfolio_url}" if profile.portfolio_url else "",
        f"- LinkedIn: {profile.linkedin_url}" if profile.linkedin_url else "",
        f"- GitHub: {profile.github_url}" if profile.github_url else "",
    ]
    links = "\n".join([l for l in links if l])

    return f"""
Goal: Apply to each job URL using the applicant data below.

JOB_URLS:
{json.dumps(job_urls, indent=2)}

APPLICANT PROFILE:
- Name: {profile.full_name}
- Email: {profile.email}
- Phone: {profile.phone}
- Location: {profile.location}
- Work authorization: {profile.work_authorization}
- Desired salary: {profile.desired_salary}
- Notice period: {profile.notice_period}
{links if links else ""}

RESUME (public URL):
{profile.resume_url}

INSTRUCTIONS:
1) Open each URL and go to the application form.
2) Upload the resume (download the URL to a temp file if the page needs a file picker).
3) Fill standard fields from the profile.
4) For free-form questions (e.g., “Why this role?”), write 2–4 concise, professional sentences using profile + page context.
5) If login/unique questions/captcha block progress, stop for that site and record the reason.

RETURN STRUCTURED OUTPUT ONLY as JSON:
{{"results": [
  {{"url": "<job-url>", "site": "<host>", "status": "submitted|needs_input|error", "details": "<short note>"}}
]}}
""".strip()


async def run_browser_use(goal: str, api_key: str, model: str, headless: bool, max_steps: int):
    """Run a browser-use Agent and return its final text output."""
    llm = OpenAIChat(api_key=api_key, model=model, temperature=0.2)

    # IMPORTANT: Pass viewport as a dict for cross-version safety.
    ctx_cfg = BrowserContextConfig(
        headless=headless,
        viewport_size={"width": 1400, "height": 900},
        # Add optional fields here: user_agent=..., proxy=..., cookies=..., geolocation=...
    )

    controller = Controller(context_config=ctx_cfg)

    agent = Agent(
        task=goal,
        llm=llm,
        controller=controller,
        max_actions=max_steps,  # guard against runaway loops
    )

    result = await agent.run()
    # Normalize across versions (.final_result vs .final_result_text)
    text = getattr(result, "final_result", None) or getattr(result, "final_result_text", "") or str(result)
    return text


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Job Copilot (browser-use)", page_icon="🧭", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🔌 LLM / Browser Settings")
    st.caption("Uses the OpenAI driver from browser-use. Provide your API key below.")
    openai_key = st.text_input("OpenAI API Key", type="password")
    model = st.text_input("Model", value="gpt-4o-mini", help="Any chat model supported by your key.")
    headless = st.checkbox("Run headless", value=False)
    max_steps = st.slider("Max actions", 10, 300, 120, 10)
    st.caption("Tip: keep headless OFF while debugging; ON for servers.")

tab_chat, tab_batch = st.tabs(["💬 Chat", "📥 Batch Apply"])


# ---------------- Chat Tab ----------------
with tab_chat:
    st.subheader("Chat to Control Your Agent")

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input(
        "e.g., 'apply to https://jobs.lever.co/... and https://boards.greenhouse.io/...'", disabled=not openai_key
    )
    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        urls = extract_urls(prompt)
        with st.chat_message("assistant"):
            if not urls:
                st.write("Please include one or more job URLs in your message.")
            else:
                st.info("Using a minimal profile—use the Batch tab for full control.")
                name = st.text_input("Full name", key="chat_name")
                email = st.text_input("Email", key="chat_email")
                phone = st.text_input("Phone", key="chat_phone")
                resume_url = st.text_input("Public resume URL", key="chat_resume_url")

                if st.button("Run", disabled=not (openai_key and name and email and phone and resume_url)):
                    try:
                        profile = Profile(full_name=name, email=email, phone=phone, resume_url=resume_url)
                        goal = build_goal(profile, urls)
                        with st.status("Running browser-use agent…", expanded=True) as s:
                            out = asyncio.run(run_browser_use(goal, openai_key, model, headless, max_steps))
                            s.update(label="Agent finished", state="complete")
                        st.code(out, language="json")
                        st.session_state.history.append({"role": "assistant", "content": out})
                    except ValidationError as ve:
                        st.error(f"Invalid profile: {ve}")
                    except Exception as e:
                        st.error(str(e))


# ---------------- Batch Tab ----------------
with tab_batch:
    st.subheader("Batch Apply")

    col1, col2 = st.columns([0.55, 0.45], gap="large")

    with col1:
        jobs_text = st.text_area(
            "Job URLs (one per line)",
            height=180,
            placeholder="https://jobs.lever.co/...\nhttps://boards.greenhouse.io/...\nhttps://careers.workday.com/...",
        )

    with col2:
        st.markdown("**Applicant Profile**")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        location = st.text_input("Location (optional)")
        portfolio_url = st.text_input("Portfolio URL (optional)")
        linkedin_url = st.text_input("LinkedIn URL (optional)")
        github_url = st.text_input("GitHub URL (optional)")
        work_auth = st.text_input("Work Authorization", value="Authorized to work in the U.S.")
        desired_salary = st.text_input("Desired Salary (optional)")
        notice_period = st.text_input("Notice Period (optional)")
        resume_url = st.text_input("Public Résumé URL (PDF)", placeholder="https://.../resume.pdf")

    can_run = all([openai_key, full_name, email, phone, resume_url])

    if st.button("🚀 Apply Now", type="primary", disabled=not can_run):
        try:
            urls = [u.strip() for u in jobs_text.splitlines() if u.strip()]
            validated = JobBatch(job_urls=urls)
            profile = Profile(
                full_name=full_name,
                email=email,
                phone=phone,
                location=location,
                portfolio_url=portfolio_url or None,
                linkedin_url=linkedin_url or None,
                github_url=github_url or None,
                work_authorization=work_auth,
                desired_salary=desired_salary,
                notice_period=notice_period,
                resume_url=resume_url,
            )
            goal = build_goal(profile, validated.job_urls)

            with st.status("Running browser-use agent…", expanded=True) as s:
                out = asyncio.run(run_browser_use(goal, openai_key, model, headless, max_steps))
                s.update(label="Agent finished", state="complete")

            # Try to render strict JSON if the agent followed instructions
            try:
                parsed = json.loads(out)
                st.success("Run complete.")
                st.json(parsed)
            except Exception:
                st.warning("Agent didn't return strict JSON; showing raw output.")
                st.code(out, language="json")

        except ValidationError as ve:
            st.error(f"Invalid input: {ve}")
        except Exception as e:
            st.error(str(e))
