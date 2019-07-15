

def test_init():
    from sasctl._services import service, model_repository
    mr = model_repository()

    t0 = model_repository.is_available()
    # print(fs)
