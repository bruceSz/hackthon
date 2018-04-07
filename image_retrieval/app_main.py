from image_retrieval.web import app

if __name__ == "__main__":
    app = app.PictureSearch()
    app.run()