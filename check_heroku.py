import requests
df = {
    "age": 30,
    "workclass": "State-gov",
    "education": "Masters",
    "maritalStatus": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 20,
    "nativeCountry": "Germany"
    }
r = requests.post('https://project-3-jan-c1491110d0b5.herokuapp.com', json=df)

assert r.status_code == 200

print("status_code: " + str(r.status_code))
print("response: " + str(r.json()))