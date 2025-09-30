from otter.test_files import test_case

OK_FORMAT = False

name = "p1"
points = 10

@test_case(points=None, hidden=False)
def test_model(model, env=globals()):
    np = env['np']
    n = 100
    X = np.random.rand(n, 2)
    X_and_1 = np.hstack((X, np.ones((n, 1))))
    bfw1 = np.random.rand(200, 300, 3)
    Yhat1 = model(X_and_1, bfw1)
    bfw2 = np.random.rand(200, 300, 3)
    Yhat2 = model(X_and_1, bfw2)
    assert np.allclose(model(X_and_1, 13 * bfw1 + 17 * bfw2), 13 * Yhat1 + 17 * Yhat2)

