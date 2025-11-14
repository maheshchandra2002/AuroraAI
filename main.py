"""
Main module for the simple question‑answering service.

This service exposes a single HTTP endpoint, ``/ask``, that accepts a natural
language question about member data (e.g. "When is Layla planning her trip to
London?") and returns a succinct answer derived from the available member
messages.  Messages are retrieved from the public ``/messages`` API provided
in the exercise description.  Answers are generated using lightweight text
matching heuristics and a fallback similarity search when a structured
response cannot be derived.

The service is built with FastAPI and can be started with Uvicorn.  See
README.md for usage instructions and design considerations.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    # dateparser is used to convert natural language dates into datetime objects.
    import dateparser  # type: ignore
except ImportError:
    dateparser = None  # dateparser is optional; fallback behaviours are provided

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except ImportError:
    TfidfVectorizer = None

# Configure a basic logger so that deployments can see what the service is doing.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Member QA Service", description="Answers natural‑language questions using member messages", version="0.1.0")

# Base URL for retrieving messages.  The trailing slash is significant.
API_URL = "https://november7-730026606190.europe-west1.run.app/messages/"


class AnswerResponse(BaseModel):
    """Structure of the API response."""

    answer: str


def fetch_all_messages() -> List[dict]:
    """Retrieve all messages from the remote API.

    The remote endpoint returns paginated results with a ``skip`` and ``limit``
    query parameter.  This function automatically pages through the data until
    all messages have been collected.  If the remote call fails at any point
    an exception is raised.

    Returns
    -------
    list of dict
        A list of message objects with keys ``id``, ``user_id``, ``user_name``,
        ``timestamp`` and ``message``.
    """
    messages: List[dict] = []
    skip = 0
    limit = 100

    while True:
        logger.debug("Fetching messages: skip=%s limit=%s", skip, limit)
        try:
            resp = requests.get(API_URL, params={"skip": skip, "limit": limit}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to retrieve messages: %s", exc)
            raise HTTPException(status_code=502, detail="Failed to retrieve messages from upstream API") from exc
        data = resp.json()
        items = data.get("items", [])
        messages.extend(items)
        skip += limit
        # When fewer than ``limit`` messages are returned we have exhausted the dataset.
        if len(items) < limit:
            break
    return messages


def detect_person(question: str, messages: List[dict]) -> Optional[str]:
    """Attempt to detect the referenced person (user_name) in the question.

    The function performs a case‑insensitive search of all user names that
    appear in the message dataset.  If the question contains a user's full
    name it is returned; otherwise ``None`` is returned.

    Parameters
    ----------
    question : str
        The input question from the client.
    messages : list of dict
        Collection of messages used to derive candidate user names.

    Returns
    -------
    Optional[str]
        The detected user name or ``None`` if no match was found.
    """
    qlower = question.lower()
    # Build a set of unique user names to minimise redundant comparisons.
    seen = set()
    for m in messages:
        user_name = m.get("user_name", "").strip()
        lower_name = user_name.lower()
        if user_name and lower_name not in seen:
            seen.add(lower_name)
            if lower_name in qlower:
                return user_name
    return None


def parse_number(text: str) -> Optional[int]:
    """Parse an integer from text where numbers may be expressed as digits or words.

    This helper recognises simple number words up to 'twelve'.  It attempts to
    convert digits directly and returns ``None`` if no interpretable number is
    found.

    Parameters
    ----------
    text : str
        Input string possibly containing a number.

    Returns
    -------
    Optional[int]
        The extracted integer value or ``None``.
    """
    # Try to parse a numeric literal first.
    digit_match = re.search(r"\b(\d+)\b", text)
    if digit_match:
        try:
            return int(digit_match.group(1))
        except ValueError:
            pass
    # Mapping of common number words to their numeric equivalents.
    WORD_NUMBERS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    for word, value in WORD_NUMBERS.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            return value
    return None


def parse_date_from_text(text: str) -> Optional[str]:
    """Extract a human‑readable date from arbitrary text.

    If the optional dependency ``dateparser`` is available it will be used
    to parse the first date found in the provided text.  Parsed dates are
    formatted as ``Month Day, Year``.  If no date can be parsed the function
    returns ``None``.

    Parameters
    ----------
    text : str
        Text potentially containing date information.

    Returns
    -------
    Optional[str]
        Formatted date string or ``None``.
    """
    if dateparser is None:
        return None
    # We instruct dateparser to prefer dates in the future because many of the
    # messages refer to upcoming events.  This is important when parsing words
    # like "next Friday" or "tomorrow" which would otherwise resolve to past
    # dates when the service runs after the fact.
    settings = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    dt = dateparser.parse(text, settings=settings)
    if dt:
        try:
            return dt.strftime("%B %d, %Y")
        except Exception:
            # If formatting fails we return ISO format as fallback.
            return dt.isoformat()
    return None


def answer_when_question(question: str, person: str, messages: List[dict]) -> Optional[str]:
    """Attempt to answer a question beginning with 'when'.

    The function looks for expressions like "trip to X" in the question and
    searches the person's messages for any entry containing both the word
    "trip" and the specified location.  If a date can be extracted from
    the matched message it will be returned in a natural language sentence.

    Parameters
    ----------
    question : str
        The original question.
    person : str
        The detected user name (may be ``None`` if none was detected).
    messages : list of dict
        Collection of messages to search.

    Returns
    -------
    Optional[str]
        A formatted answer or ``None`` if no suitable message is found.
    """
    # Identify the location from phrases like "trip to Paris" or "trip in London".
    loc_match = re.search(r"\btrip\s+(?:to|in)\s+([A-Za-z\s]+)", question, re.IGNORECASE)
    location = loc_match.group(1).strip() if loc_match else None
    candidate_msgs = messages
    if person:
        candidate_msgs = [m for m in messages if m.get("user_name") == person]
    # Search messages for the relevant event.
    for msg in candidate_msgs:
        mtext = msg.get("message", "")
        if "trip" in mtext.lower():
            if location and location.lower() not in mtext.lower():
                continue
            date_str = parse_date_from_text(mtext)
            if date_str:
                # Compose a friendly answer.
                target_name = person if person else msg.get("user_name", "The user")
                loc_part = f" to {location}" if location else ""
                return f"{target_name} is planning a trip{loc_part} on {date_str}."
    return None


def answer_how_many_question(question: str, person: str, messages: List[dict]) -> Optional[str]:
    """Answer questions of the form "how many X does user have".

    Currently the system only supports answering queries about the number of cars.
    It searches the user's messages for statements referencing cars and
    extracts the first integer from the sentence.  If a number cannot be found
    ``None`` is returned.
    """
    if not person:
        return None
    # Determine the target item (e.g. cars, houses) from the question.
    item_match = re.search(r"how many\s+([a-z]+)", question, re.IGNORECASE)
    item = item_match.group(1).lower() if item_match else ""
    for msg in messages:
        if msg.get("user_name") != person:
            continue
        text = msg.get("message", "")
        if item and item in text.lower():
            number = parse_number(text)
            if number is not None:
                return f"{person} has {number} {item}{'' if number == 1 else 's'}."
    return None


def answer_favorite_question(question: str, person: str, messages: List[dict]) -> Optional[str]:
    """Answer questions asking about someone's favourite things (e.g. restaurants).

    The system looks for messages containing both the words "favorite" and
    "restaurant" (or the plural form) and attempts to extract a list of
    restaurants from the text.  If found, the list is returned verbatim.
    """
    if not person:
        return None
    for msg in messages:
        if msg.get("user_name") != person:
            continue
        text = msg.get("message", "")
        lower = text.lower()
        if "favorite" in lower and "restaurant" in lower:
            # Attempt to extract a list of restaurants after a colon or dash.
            match = re.search(r"favorite[\w\s]*restaurants?[^:\-]*[:\-]\s*([^\.]+)", text, re.IGNORECASE)
            if match:
                restaurants = match.group(1).strip()
                return f"{person}'s favorite restaurants are {restaurants}."
            # Fallback: return the entire sentence if extraction fails.
            return f"{person} mentioned: {text}"
    return None


def fallback_similarity(question: str, messages: List[dict]) -> str:
    """Return the message most similar to the question using a TF‑IDF model.

    When the system cannot provide a structured answer to the question it
    computes cosine similarity between the question and every message in the
    dataset (using TF‑IDF vectors).  The text of the most similar message is
    returned as the answer.  If scikit‑learn is not installed a generic
    apology is returned instead.
    """
    if TfidfVectorizer is None:
        return "Sorry, I'm unable to answer that question right now."
    # Build the corpus with the question followed by all messages.
    corpus = [question] + [m.get("message", "") for m in messages]
    # Compute the TF‑IDF matrix.  We cap max_features to limit memory usage.
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(corpus)
    # Compute similarity of the question vector against all other vectors.
    # The first row corresponds to the question.
    query_vec = matrix[0]
    rest = matrix[1:]
    # Cosine similarity reduces to dot product because TF‑IDF vectors are normalised.
    similarities = (rest @ query_vec.T).toarray().flatten()
    # Identify the index of the most similar message.
    idx = int(similarities.argmax()) if similarities.size else 0
    best_message = messages[idx].get("message", "")
    return best_message


@app.get("/ask", response_model=AnswerResponse)
def ask(question: str = Query(..., description="Your natural‑language question")) -> AnswerResponse:
    """API endpoint to answer questions about member data.

    The query parameter ``question`` should contain a natural language sentence
    referencing a member by name and asking for some piece of information.
    This handler will fetch all messages from the upstream API, detect the
    referenced person, and route the question to a specialised handler.  If
    no structured answer is produced a generic fallback will be returned.
    """
    logger.info("Received question: %s", question)
    # Retrieve all messages.  In a production system this could be cached and
    # refreshed periodically to improve latency.
    messages = fetch_all_messages()
    # Attempt to detect which member is being referenced.
    person = detect_person(question, messages)
    # Determine question type and delegate to the appropriate handler.
    qlower = question.strip().lower()
    answer: Optional[str] = None
    if qlower.startswith("when"):
        answer = answer_when_question(question, person, messages)
    elif qlower.startswith("how many"):
        answer = answer_how_many_question(question, person, messages)
    elif "favorite" in qlower and "restaurant" in qlower:
        answer = answer_favorite_question(question, person, messages)
    # If no specific answer was produced use the similarity fallback.
    if not answer:
        answer = fallback_similarity(question, messages)
    return AnswerResponse(answer=answer)
