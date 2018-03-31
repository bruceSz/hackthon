from flask import Flask, request, render_template
from module import search_engine


class PictureSearch(object):

    APP = Flask(__name__)

    def __init__(self):
        self.se_ = search_engine


    @APP.route("/")
    def index(self):
        return render_template("index.html")

    @APP.route("/find_similar_pic",methods=['POST'])
    def search(self):
        if request.files.get("file"):
            file = request.files.get('file')
            data = file.read()
            results =