from database import Base, engine, SessionLocal, Customer

Base.metadata.create_all(bind=engine)

db = SessionLocal()
sample_data = [
    Customer(name="Alice", gender="Female", location="Mumbai"),
    Customer(name="Bob", gender="Male", location="Delhi"),
    Customer(name="Clara", gender="Female", location="Mumbai"),
    Customer(name="David", gender="Male", location="Pune"),
    Customer(name="Eva", gender="Female", location="Chennai"),
]

db.add_all(sample_data)
db.commit()
db.close()
