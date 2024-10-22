from fastapi import FastAPI, WebSocket, HTTPException, Request
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dateutil import parser
from typing import Optional, List
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

# Import from your psychological analysis module
from Tingle_Brain_A_Agent_One import (nlp_model, expanded_dict, calculate_daily_scores, follow_up_questions,
                                      track_score_changes, accumulate_scores, get_top_traits, determine_tone, prepare_context_for_agent_1, 
                                      prepare_context_for_agent_2,get_analysis_from_chatgpt)
# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client['mydatabase']
collection = db['entries']

# FastAPI application
app = FastAPI()

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Entry(BaseModel):
    user_id: str                                        
    title: str
    content: str
    time: Optional[str] = None

@app.get("/")
def root():
    # Serve the index.html file
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Log any MongoDB-related operations
logging.basicConfig(level=logging.DEBUG)

@app.post("/submit_entry/")
def submit_entry(entry: Entry):
    try:
        logging.debug("Inserting entry into MongoDB")
        # Parse time or use current date if not provided
        entry_time = parser.parse(entry.time).strftime("%Y-%m-%d") if entry.time else datetime.now().strftime("%Y-%m-%d")

        # Insert entry into MongoDB
        result = collection.insert_one({
            "user_id": entry.user_id,
            "title": entry.title,
            "content": entry.content,
            "time": entry_time
        })

        if result:
            logging.debug(f"Entry inserted with ID: {str(result.inserted_id)} for user: {entry.user_id}")

            # Fetch diary entries from MongoDB
            diary_entries = fetch_diary_entries(entry.user_id)
            
            # Calculate daily scores
            daily_scores = calculate_daily_scores(diary_entries, nlp_model, expanded_dict)

            # Get the accumulated scores and top traits
            accumulated_scores = accumulate_scores(daily_scores)
            top_traits = get_top_traits(accumulated_scores)

            # Determine the tone based on the latest diary entry
            latest_entry_date = max(diary_entries.keys())
            tone = determine_tone(daily_scores[latest_entry_date])

            # Prepare the context for context_agent_1 (initial psychological analysis)
            context_agent_1 = prepare_context_for_agent_1(diary_entries, daily_scores, accumulated_scores, top_traits)
            analysis = get_analysis_from_chatgpt(context_agent_1)

            # Prepare the context for context_agent_2 (follow-up conversation)
            context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits, tone)

            # Optionally, trigger the follow-up question interaction
            follow_up_questions(analysis, diary_entries, daily_scores, None, accumulated_scores, top_traits, tone)

            # Return the analysis and success message
            return {"message": "Entry submitted and analyzed successfully", "analysis": analysis}
        else:
            logging.error("Entry submission failed")
            raise HTTPException(status_code=500, detail="Entry submission failed")
    
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        # Handle exceptions gracefully with a detailed message
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/data/{user_id}")
def get_data(user_id: str):
    try:
        data = list(collection.find({"user_id": user_id}, {"_id": 0}))  # Exclude MongoDB _id from the result
        if not data:
            return {"message": "No data found for user"}
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# WebSocket for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to the Tingle chat. You can now ask follow-up questions.")
    
    while True:
        data = await websocket.receive_text()
        if data.lower() == "exit":
            await websocket.send_text("Goodbye!")
            break

        # Placeholder response for follow-up questions
        response = f"Received: {data}. Analysis will be provided soon."
        await websocket.send_text(response)

# Function to fetch diary entries from MongoDB
def fetch_diary_entries(user_id: str):
    entries = collection.find({"user_id": user_id})
    diary_entries = {}
    for entry in entries:
        diary_entries[datetime.strptime(entry['time'], '%Y-%m-%d').date()] = [entry['content']]
    return diary_entries

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# access the FastAPI app at http://127.0.0.1:8000/ and MongoDB data via http://127.0.0.1:8000/data.
# http://127.0.0.1:8000/docs 
# conda activate my_env
# uvicorn main:app --reload
