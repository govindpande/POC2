import asyncio
import json
import os
import re
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st
from pydantic import BaseModel, Field, HttpUrl, ValidationError

# Skyvern Python SDK
# Docs (Quickstart & SDK): https://docs.skyvern.com/getting-started/quickstart
# Run Tasks (params like engine, data_extraction_schema, webhook_url, max_steps, proxy_location, TOTP, browser_session): https://docs.skyvern.com/running-tasks/run-tasks
# Browser Sessions: https://docs.skyvern.com/browser-sessions/introduction
from skyvern import Skyvern


# -----------------------
# App Config
# -----------------------
st.set_page_config(
    page_title="Job Application Copilot (Skyvern + Streamlit)",
    page_icon="🧑‍💻",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "browser_session_id" not in st.session_state:
    st.session_state.browser_session_id = None

if "last_runs" not in st.session_state:
    # store recent Skyvern run IDs: [{"run_id": "...", "prompt": "..."}]
    st.session_state.last_runs = []


# -----------------------
# Models
# -----------------------
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
    # Either a local file path (if running locally) OR a public URL if using Skyvern Cloud
    resume_path_or_url: str = Field(..., description="Local file path or public URL to the résumé")


class JobBatch(BaseModel):
    job_urls: List[HttpUrl]


# -----------------------
# Helpers
# -----------------------
URL_REGEX = r"(https?://[^\s]+)"


def extract_urls(text: str) -> List[str]:
    return re.findall(URL_REGEX, text or "")


def build_application_prompt(profile: Profile, job_urls: List[str]) -> str:
    """
    A clear, deterministic prompt that asks Skyvern to iterate through URLs and apply.
    """
    links = [
        f"- Portfolio: {profile.portfolio_url}" if profile.portfolio_url else "",
        f"- LinkedIn: {profile.linkedin_url}" if profile.linkedin_url else "",
        f"- GitHub: {profile.github_url}" if profile.github_url else "",
    ]
    links = "\n".join([l for l in links if l])

    prompt = f"""
You are an expert job-application agent.

OBJECTIVE
- For each URL in JOB_URLS, navigate to the job application page and submit an application using the provided PROFILE. 
- Upload the résumé from RESUME_SOURCE when prompted.
- Answer standard questions from PROFILE. 
- For free-form questions (e.g., “Why this role?”), write short, professional, tailored answers (2–4 sentences) using PROFILE details.

JOB_URLS
{json.dumps(job_urls, indent=2)}

PROFILE
- Full name: {profile.full_name}
- Email: {profile.email}
- Phone: {profile.phone}
- Location: {profile.location}
- Work authorization: {profile.work_authorization}
- Desired salary: {profile.desired_salary}
- Notice period: {profile.notice_period}
{links if links else ""}

RESUME_SOURCE
{profile.resume_path_or_url}

RULES
- If login is required, use credentials stored in Skyvern (by domain) if available. If 2FA appears, use configured TOTP support as available.
- If an application requires answers you cannot infer from PROFILE OR a blocker occurs (hard captcha, missing auth), STOP for that URL and mark status "needs_input" with a short description.
- After each attempt, record a structured result as per the schema (do not include any extra keys).
- COMPLETE after attempting all URLs or encountering blockers for all.

OUTPUT
Return a JSON object named "results" (array) where each item is:
  {{
    "url": "<the job URL>",
    "site": "<host, e.g., lever.co>",
    "status": "<one of: submitted | needs_input | error>",
    "details": "<short note like 'submitted - confirmation page seen' or 'needs_input - unique essay question'>"
  }}

When done, return ONLY that JSON object. 
"""
    return prompt.strip()


def build_output_schema() -> Dict[str, Any]:
    """
    A JSON Schema to make Skyvern return consistent results via the data_extraction_schema param.
    """
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "site": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["submitted", "needs_input", "error"],
                        },
                        "details": {"type": "string"},
                    },
                    "required": ["url", "status"],
                },
            }
        },
        "required": ["results"],
    }


def run_sync(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def ensure_local_file_for_upload(upload) -> Optional[str]:
    """
    Save an uploaded file to a temp path and return the path.
    Useful when running Skyvern locally (so the browser can access the file).
    """
    if upload is None:
        return None
    suffix = os.path.splitext(upload.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return tmp.name


# -----------------------
# Sidebar: Connection + Settings
# -----------------------
with st.sidebar:
    st.header("🔌 Skyvern Connection")

    api_key = st.text_input("Skyvern API Key", type="password", help="Find it in app.skyvern.com → Settings")
    base_url = st.text_input("Base URL (optional)", placeholder="https://api.skyvern.com", help="Leave blank for default cloud")

    st.divider()
    st.subheader("Agent Settings")

    engine = st.selectbox(
        "Engine",
        options=["skyvern-2.0", "skyvern-1.0", "openai-cua", "anthropic-cua", "ui-tars"],
        index=0,
        help="Default skyvern-2.0 works well with complex, multi-step tasks.",
    )
    max_steps = st.number_input("Max steps", min_value=10, max_value=400, value=120, step=10,
                                help="Agent fails after exceeding this; useful to control cost.")
    proxy_location = st.text_input("Proxy location (optional)", placeholder="us, eu, in ...")
    webhook_url = st.text_input("Webhook URL (optional)", placeholder="https://your-server.com/skyvern-webhook")

    st.subheader("2FA / TOTP (optional)")
    totp_identifier = st.text_input("TOTP Identifier (optional)", help="Used to associate codes with runs")
    totp_url = st.text_input("TOTP URL (optional)", help="Skyvern fetches TOTP codes from here if needed")

    st.subheader("Browser Session")
    col_bs1, col_bs2 = st.columns(2)
    with col_bs1:
        if st.button("Create Browser Session"):
            client = Skyvern(api_key=api_key, base_url=base_url or None)
            session = run_sync(client.create_browser_session(timeout=60))
            st.session_state.browser_session_id = session.browser_session_id
            st.success(f"Created: {session.browser_session_id}")
    with col_bs2:
        if st.button("Close Browser Session", disabled=not st.session_state.browser_session_id):
            client = Skyvern(api_key=api_key, base_url=base_url or None)
            if st.session_state.browser_session_id:
                run_sync(client.close_browser_session(st.session_state.browser_session_id))
                st.success(f"Closed: {st.session_state.browser_session_id}")
                st.session_state.browser_session_id = None

    if st.session_state.browser_session_id:
        st.caption(f"Active session: {st.session_state.browser_session_id}")

    st.divider()
    st.markdown(
        "ℹ️ **Notes**: You can pass `wait_for_completion=True` to block until the task finishes, "
        "or set a `webhook_url` and poll later with the **Get Run** API. Also, you can persist login with a Browser Session. "
    )


# -----------------------
# Tabs: Chat & Batch
# -----------------------
tab_chat, tab_batch = st.tabs(["💬 Chat", "📥 Batch Apply"])


# -----------------------
# CHAT TAB
# -----------------------
with tab_chat:
    st.subheader("Chat to Control Your Job Copilot")

    # render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Tell me what to do, e.g. 'apply to https://jobs.lever.co/...' or 'status tsk_...'")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Branch: status check
        status_match = re.match(r"^\s*status\s+(tsk_[A-Za-z0-9]+|wr_[A-Za-z0-9]+)\s*$", user_input.strip(), re.I)
        urls = extract_urls(user_input)

        if status_match:
            run_id = status_match.group(1)
            client = Skyvern(api_key=api_key, base_url=base_url or None)
            with st.chat_message("assistant"):
                with st.status("Fetching run status...", expanded=False) as s:
                    try:
                        run = run_sync(client.get_run(run_id))
                        s.update(label="Run status fetched", state="complete")
                        st.write(run.model_dump())
                    except Exception as e:
                        s.update(label="Failed to fetch run", state="error")
                        st.error(str(e))

        elif urls:
            # Minimal form ask for resume URL if none set in batch tab
            with st.chat_message("assistant"):
                st.write("Nice—I'll prepare an application run for these URLs.")
                # Quick ad-hoc profile
                st.info("Using a minimal in-chat profile. For full control, use **Batch Apply** tab.")
                name = st.text_input("Full name", key="chat_name")
                email = st.text_input("Email", key="chat_email")
                phone = st.text_input("Phone", key="chat_phone")
                resume_url = st.text_input("Public Résumé URL (required for Skyvern Cloud)", key="chat_resume_url")

                if st.button("Run Application Task", disabled=not (api_key and name and email and phone and resume_url)):
                    try:
                        profile = Profile(
                            full_name=name,
                            email=email,
                            phone=phone,
                            resume_path_or_url=resume_url,
                        )
                        prompt = build_application_prompt(profile, urls)
                        schema = build_output_schema()

                        client = Skyvern(api_key=api_key, base_url=base_url or None)

                        with st.status("Submitting task to Skyvern…", expanded=False) as s:
                            task = run_sync(
                                client.run_task(
                                    prompt=prompt,
                                    engine=engine,
                                    data_extraction_schema=schema,
                                    max_steps=max_steps,
                                    proxy_location=proxy_location or None,
                                    webhook_url=webhook_url or None,
                                    totp_identifier=totp_identifier or None,
                                    totp_url=totp_url or None,
                                    browser_session_id=st.session_state.browser_session_id,
                                    wait_for_completion=True,  # block and show result
                                )
                            )
                            s.update(label="Task finished", state="complete")

                        st.session_state.last_runs.append({"run_id": task.run_id, "prompt": prompt})
                        st.success(f"Run complete: **{task.run_id}**")
                        st.json(task.output or {})
                    except ValidationError as ve:
                        st.error(f"Profile invalid: {ve}")
                    except Exception as e:
                        st.error(str(e))
        else:
            with st.chat_message("assistant"):
                st.write("I can apply to jobs if you paste URLs, or check a run with `status tsk_...`")


# -----------------------
# BATCH APPLY TAB
# -----------------------
with tab_batch:
    st.subheader("Batch Apply to Multiple Job URLs")

    col_left, col_right = st.columns([0.55, 0.45], gap="large")

    with col_left:
        st.markdown("**Job URLs (one per line)**")
        job_urls_text = st.text_area(
            "Paste job links (Lever, Greenhouse, Workday, Taleo, etc.)",
            height=180,
            placeholder="https://jobs.lever.co/...\nhttps://boards.greenhouse.io/...\nhttps://... ",
        )

        uploaded_resume = st.file_uploader("Upload résumé (use this if you're running Skyvern locally)", type=["pdf", "doc", "docx"])
        resume_public_url = st.text_input("OR provide a public Résumé URL (required if using Skyvern Cloud)")

        # If user uploads a file, save to temp path
        resume_local_path = ensure_local_file_for_upload(uploaded_resume) if uploaded_resume else None
        # prefer URL if both provided
        resume_source = resume_public_url or (resume_local_path or "")

    with col_right:
        st.markdown("**Your Profile**")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        location = st.text_input("Location", placeholder="City, State")
        portfolio_url = st.text_input("Portfolio URL (optional)")
        linkedin_url = st.text_input("LinkedIn URL (optional)")
        github_url = st.text_input("GitHub URL (optional)")
        work_auth = st.text_input("Work Authorization", value="Authorized to work in the U.S.")
        desired_salary = st.text_input("Desired Salary (optional)", placeholder="e.g., $150k base or market rate")
        notice_period = st.text_input("Notice Period (optional)", placeholder="e.g., 2 weeks")

        can_run = all([api_key, full_name, email, phone, resume_source])

        if st.button("🚀 Apply Now", type="primary", disabled=not can_run):
            try:
                urls = [u.strip() for u in job_urls_text.splitlines() if u.strip()]
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
                    resume_path_or_url=resume_source,
                )

                prompt = build_application_prompt(profile, validated.job_urls)
                schema = build_output_schema()

                client = Skyvern(api_key=api_key, base_url=base_url or None)

                with st.status("Submitting task to Skyvern…", expanded=True) as s:
                    task = run_sync(
                        client.run_task(
                            prompt=prompt,
                            engine=engine,
                            data_extraction_schema=schema,
                            max_steps=max_steps,
                            proxy_location=proxy_location or None,
                            webhook_url=webhook_url or None,
                            totp_identifier=totp_identifier or None,
                            totp_url=totp_url or None,
                            browser_session_id=st.session_state.browser_session_id,
                            wait_for_completion=True,  # block; alternatively use webhook + poll
                        )
                    )
                    s.update(label="Task finished", state="complete")

                st.session_state.last_runs.append({"run_id": task.run_id, "prompt": prompt})

                st.success(f"Run complete: **{task.run_id}**")
                if task.output:
                    try:
                        st.json(task.output)
                    except Exception:
                        st.write(task.output)
                else:
                    st.info("No structured output returned. Check the run in Skyvern’s visualizer.")

            except ValidationError as ve:
                st.error(f"Invalid input: {ve}")
            except Exception as e:
                st.error(str(e))

    st.divider()

    if st.session_state.last_runs:
        st.markdown("### Recent Runs")
        for r in reversed(st.session_state.last_runs[-5:]):
            st.code(r["run_id"], language="text")
