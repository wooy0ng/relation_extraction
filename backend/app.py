import sys
import os
import uvicorn

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# import ..code.inference
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
child_dir = os.path.join(parent_dir, 'code')

sys.path.append(parent_dir)
sys.path.append(child_dir)

from mlproject.inference import inference_one


app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class UserInput(BaseModel):
    '''
    example:
        {
            sentence: "sentence",
            subject: "subject",
            object: "object"
        }
    '''
    sentence: str
    subject: str
    object: str


@app.post('/user_input', description="get user input from webpage")
def get_user_input(req: UserInput):
    args = {
        'sentence': req.sentence,
        'subject_entity': req.subject,
        'object_entity': req.object
    }
    pred = inference_one(args)
    json_res = jsonable_encoder({'inference': pred})
    
    return JSONResponse(content=json_res)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)