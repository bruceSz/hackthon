def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


@Singleton
class A(object):
    def __init__(self):
        print("Init A.")


def main():
    print("Begin init test")

    a = A()
    print("After init the first one")
    b = A()
    print("Init finished.")



if __name__ == "__main__":
    main()