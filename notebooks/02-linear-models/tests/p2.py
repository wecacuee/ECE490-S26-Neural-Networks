from otter.test_files import test_case

OK_FORMAT = False

name = "p2"
points = 10

@test_case(points=None, hidden=False)
def test_grad_loss(grad_loss, env=globals()):
    np = env['np']
    partial = env['partial']
    numerical_jacobian = env['numerical_jacobian']
    check_numerical_jacobian = env['check_numerical_jacobian']
    loss = env['loss']
    n = 100
    X = np.random.rand(n, 2)
    X_and_1 = np.hstack((X, np.ones((n, 1))))
    Y = np.random.randint(0, 1, size=n) * 2 - 1
    check_numerical_jacobian(partial(loss, X_and_1, Y), partial(grad_loss, X_and_1, Y), nD=3)

