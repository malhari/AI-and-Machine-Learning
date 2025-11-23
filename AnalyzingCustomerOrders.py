# ---------------------------------------------
# 1. STORE CUSTOMER ORDERS
# ---------------------------------------------

# List of customer names
customers = ["Alice", "Bob", "Charlie", "David", "Emma"]

# List of orders → (customer, product, price, category)
orders = [
    ("Alice", "Laptop", 900, "Electronics"),
    ("Alice", "Headphones", 50, "Electronics"),
    ("Bob", "T-Shirt", 25, "Clothing"),
    ("Bob", "Jeans", 40, "Clothing"),
    ("Charlie", "Mixer", 70, "Home Essentials"),
    ("Charlie", "T-Shirt", 25, "Clothing"),
    ("David", "Smartphone", 600, "Electronics"),
    ("Emma", "Blender", 45, "Home Essentials"),
    ("Emma", "Jeans", 40, "Clothing"),
]

# Dictionary: customer → list of products purchased
customer_orders = {}

for cust, product, price, category in orders:
    customer_orders.setdefault(cust, []).append((product, price, category))


# ---------------------------------------------
# 2. CLASSIFY PRODUCTS BY CATEGORY
# ---------------------------------------------

product_to_category = {product: category for _, product, _, category in orders}

# Unique product categories
unique_categories = set(category for _, _, _, category in orders)

print("\nAvailable Product Categories:")
print(unique_categories)


# ---------------------------------------------
# 3. ANALYZE CUSTOMER ORDERS
# ---------------------------------------------

customer_total = {}
customer_classification = {}

for cust, items in customer_orders.items():
    total = sum(price for _, price, _ in items)
    customer_total[cust] = total

    # classify
    if total > 100:
        customer_classification[cust] = "High-value Buyer"
    elif 50 <= total <= 100:
        customer_classification[cust] = "Moderate Buyer"
    else:
        customer_classification[cust] = "Low-value Buyer"


# ---------------------------------------------
# 4. GENERATE BUSINESS INSIGHTS
# ---------------------------------------------

# Revenue per category
category_revenue = {}

for _, product, price, category in orders:
    category_revenue[category] = category_revenue.get(category, 0) + price

# Unique products across all orders
unique_products = set(product for _, product, _, _ in orders)

# Customers who purchased Electronics
electronics_buyers = [
    cust for cust, items in customer_orders.items()
    if any(category == "Electronics" for _, _, category in items)
]

# Top 3 highest spending customers
top_spenders = sorted(customer_total.items(), key=lambda x: x[1], reverse=True)[:3]


# ---------------------------------------------
# 5. ORGANIZE & DISPLAY DATA
# ---------------------------------------------

print("\n--- CUSTOMER SUMMARY ---")
for cust in customers:
    print(f"{cust}: Total = ${customer_total.get(cust, 0)} | {customer_classification.get(cust)}")

# Customers who purchased from multiple categories
customer_categories = {
    cust: set(category for _, _, category in items)
    for cust, items in customer_orders.items()
}

multi_category_customers = [cust for cust, cats in customer_categories.items() if len(cats) > 1]

# Customers who bought both Electronics & Clothing
electronics_customers = {cust for cust, cats in customer_categories.items() if "Electronics" in cats}
clothing_customers = {cust for cust, cats in customer_categories.items() if "Clothing" in cats}

common_ec_customers = electronics_customers & clothing_customers


# ---------------------------------------------
# 6. PRINT INSIGHTS
# ---------------------------------------------

print("\n--- BUSINESS INSIGHTS ---")
print("Revenue per Category:", category_revenue)
print("Unique Products:", unique_products)
print("Electronics Buyers:", electronics_buyers)
print("Top 3 Spenders:", top_spenders)
print("Customers with Multi-category Purchases:", multi_category_customers)
print("Customers who bought Electronics & Clothing:", list(common_ec_customers))

