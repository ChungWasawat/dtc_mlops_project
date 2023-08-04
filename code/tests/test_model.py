"""To do unit testing"""
# import model_deployment


def test_prepare_features():
    """this function can return as expected"""
    test_data = {
        "destination": ["No Urgent Place", "Home", "Work"],
        "weather": ["Sunny", "Rainy", "Snowy"],
        "time": ["10AM", "10PM", "7AM"],
        "coupon": ["Coffee House", "Coffee House", "Coffee House"],
        "expiration": ["2h", "2h", "1d"],
        "same_direction": [0, 1, 1],
        "coupon_accepting": [0, 0, 0],
    }
    print(test_data)
