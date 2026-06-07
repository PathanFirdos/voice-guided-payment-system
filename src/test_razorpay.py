import razorpay

KEY_ID = "rzp_test_SxWaDPxq47Yk8H"
KEY_SECRET = "NwSUUZUO9Vm7xc4AXFT3Fuyj"

client = razorpay.Client(
    auth=(KEY_ID, KEY_SECRET)
)

order = client.order.create({
    "amount": 10000,      # ₹100
    "currency": "INR",
    "receipt": "vispay_demo"
})

print("\nRazorpay Connected Successfully!")
print("Order ID:", order["id"])
print("Amount:", order["amount"] / 100)
print("Status:", order["status"])