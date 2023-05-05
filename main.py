from flask import Flask
from app import views
#Webserver gateway interface(WSGI
app = Flask(__name__)

app.add_url_rule(rule='/',endpoint='Home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='APP',view_func=views.app)
app.add_url_rule(rule='/app/gender_prediction/',endpoint='Gender Prediction',view_func=views.gender_prediction,methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)


