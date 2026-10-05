from streamlit.testing.v1 import AppTest
pages = ["📊 Overview","🔍 Player Explorer","💰 Price Predictor","🏆 Top Valuable Players","🏟️ Team & Season Analysis","🧪 Model Performance"]
bad = 0
for p in pages:
    at = AppTest.from_file("app.py", default_timeout=90).run()
    at.sidebar.radio[0].set_value(p).run()
    status = "OK" if not at.exception else "FAIL"
    bad += bool(at.exception)
    print(f"{status:5}{p}  charts/tables seen")
    for e in at.exception: print("   ", e.value[:400])
# predictor: new player mode, without previous price
at = AppTest.from_file("app.py", default_timeout=90).run()
at.sidebar.radio[0].set_value("💰 Price Predictor").run()
at.radio[0].set_value("A new player (enter manually)").run()
print("new-player mode:", "FAIL" if at.exception else "OK")
for e in at.exception: print("   ", e.value[:400])
# team + player variants
at = AppTest.from_file("app.py", default_timeout=90).run()
at.sidebar.radio[0].set_value("🔍 Player Explorer").run()
for nm in ["Virat Kohli","A Kamboj","Karan Sharma","Mayank Dagar","Jasprit Bumrah"]:
    try:
        at.selectbox[0].set_value(nm).run(); print(nm, "FAIL" if at.exception else "OK")
    except Exception as ex: print(nm, "skip:", str(ex)[:80])
print("failures:", bad)
