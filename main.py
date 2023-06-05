from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union



app = FastAPI()

class SalaryCalc(BaseModel):
    salary: Union[float, int]
    bonus: Union[float, int]
    taxes: Union[float, int]

@app.post("/api/calc")
def calc(input_data: SalaryCalc):
    try:
        result = input_data.salary + input_data.bonus - input_data.taxes
        return {"result": result}
    except ValueError:
        raise HTTPException(status_code=400, detail="Expected numbers, got strings.")
    except TypeError:
        missing_fields = [field for field, value in input_data.dict().items() if value is None]
        if missing_fields:
            missed_field = ', '.join(missing_fields)
            error = f"3 fields expected (salary, bonus, taxes). You forgot: {missed_field}."
            raise HTTPException(status_code=400, detail=error)
        else:
            raise HTTPException(status_code=400, detail="An error occurred.")



