"""send a request to the web service and get a response"""
import requests

coupon_receiver = {
    "destination": ["No Urgent Place", "Home", "Work"],
    "weather": ["Sunny", "Rainy", "Snowy"],
    "time": ["10AM", "10PM", "7AM"],
    "coupon": ["Coffee House", "Coffee House", "Coffee House"],
    "expiration": ["2h", "2h", "1d"],
    "same_direction": [0, 1, 1],
    "coupon_accepting": [0, 0, 0],
}

URL = "http://localhost:9696/predict"
response = requests.post(URL, json=coupon_receiver, timeout=3)  # timeout (seconds)
print(response.json())
