from flask import Flask, request, render_template
from image_retrieval.mod import search_engine


def PictureSearch():

    APP = Flask(__name__)

    se_ = search_engine.SearchEngine()


    @APP.route("/")
    def index():
        return render_template("img_retrieval/index.html")

    @APP.route("/find_similar_pic",methods=['POST'])
    def search():
        if request.files.get("file"):
            file = request.files.get("file")
            img = file.read()
            import io
            imgf = io.BytesIO(img)
            similar_img_list = se_.search(imgf)
            print(similar_img_list)
            top_1_img = similar_img_list[0]
            name = "0002.jpg"
            # TODO. ignore them now.
            #ret = render_template("img_retrieval/result.html",file_n='static/web/'+top_1_img)
            ret = render_template("img_retrieval/result.html", file_n='static/web/'+top_1_img)
            print(ret)
            return ret

    @APP.route("/api/find_similar_pic",methods = ['POST'])
    def search_api():
        if request.files.get("file"):
            file = request.files.get("file")
            img  = file.read()
            import StringIO
            img_f_path = StringIO.StringIO(img)
            similar_img_list = se_.search(img_f_path)
            print(similar_img_list)


    return APP


class PictureClassify(object):
    def __init__(self):
        self.classifier_ = None
        pass

    APP = Flask(__name__)

    @APP.route("/")
    def index(self):
        return render_template("./web/index.html")

    @APP.route("/classify",methods=['POST'])
    def classify(self):
        if request.files.get('file'):
            file = request.files.get("file")
            data = file.read()
            results = self.classifier_.classify(data)
            return render_template("./web/result.html",)
        else:
            return self.index()


def  ExampleApi():
    APP = Flask(__name__)

    @APP.route("/")
    def index():
        return render_template('example/index.html')

    @APP.route("/api")
    def index_api():
        return "{}"


    @APP.route("/count_word",methods=['POST'])
    def count():
        print("Entering count method")
        if request.files.get("file"):
            print("Got the file")
            file = request.files.get("file")
            data = file.read()
            length = len(data)
            return render_template("example/result.html",word_count=length)

    @APP.route("/api/count_word",methods=['POST'])
    def count_api():
        print("count api.")
        if request.files.get("file"):
            print("Got the file")
            file = request.files.get("file")
            data = file.read()
            length = len(data)

            return '{"word_count":'+ str(length) +'}'

        return "{ should post with file.}"

    return APP




