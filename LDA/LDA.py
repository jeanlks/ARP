# imports go here
import numpy as np
import matplotlib.pyplot as plt


# classes go here
# functions go here
def generate_points(means, covariances, num_points):
    points = list()
    for ind, n in enumerate(num_points):
        points.append(means[:, ind] + \
                      np.linalg.cholesky(np.matrix(covariances[:, :, ind])) * \
                      np.matrix(np.random.randn(2, n)))
    points = np.hstack(tuple(points))
    return points


def lda(x, mu, sigma, theta):
    x = np.matrix(x)
    xt = x.transpose()
    mu = np.matrix(mu)
    mut = mu.transpose()
    sigma = np.matrix(sigma)
    sig_inv = np.linalg.inv(sigma)
    return xt * sig_inv * mu - .5 * mut * sig_inv * mu + np.log(theta)


def qda(x, mu, sigma, theta):
    x = np.matrix(x)
    xt = x.transpose()
    mu = np.matrix(mu)
    mut = mu.transpose()
    sigma = np.matrix(sigma)
    sig_inv = np.linalg.inv(sigma)
    return -.5 * np.log(np.linalg.det(sigma)) - .5 * (xt - mut) * sig_inv * (x - mu) + np.log(theta)


def lq_classify(points, means, covariances, theta, l_or_q=0):
    groups = np.zeros(points.shape[1])
    _, k = means.shape
    _, n = points.shape
    cls_fxns = [lda, qda]
    lda_cov = np.zeros(covariances.shape)
    for ind in xrange(k):
        lda_cov[:, :, ind] = np.mean(covariances, 2)
    covs = [lda_cov, covariances]
    score_holder = np.zeros(k)
    for pt_ind in xrange(n):
        for cls_ind in xrange(k):
            score_holder[cls_ind] = cls_fxns[l_or_q](points[:, pt_ind], \
                                                     means[:, cls_ind], \
                                                     covs[l_or_q][:, :, cls_ind], \
                                                     theta[cls_ind])
        print
        score_holder
        groups[pt_ind] = np.argmax(score_holder)
    return groups


def plot_classifs(groups, points, line_specs, **kwargs):
    k = len(line_specs)
    plt.figure()
    plt.title(kwargs['the_title'])
    for cls_ind in xrange(k):
        good_inds = groups == cls_ind
        plt.plot(points[0, good_inds], points[1, good_inds], line_specs[cls_ind])
    plt.savefig(kwargs['filename'])
    plt.close()
    return 0


# main part goes here
if __name__ == '__main__':
    # decide on 5 means and 5 covariances
    k = 5
    num_points = np.round(np.linspace(25, 50, 5))
    actual_groups = list()
    for ind, pt_entry in enumerate(num_points):
        actual_groups.append(ind * np.ones(pt_entry))
    actual_groups = np.hstack(tuple(actual_groups))
    priors = num_points / np.sum(num_points)
    means = np.matrix([[1, 0, -1, -.5, .5], [1, 1, 1, -1, -1]])
    line_specs = ['*b', 'xr', 'og', '^c', 'pm']
    covariances = np.zeros((2, 2, 5))
    for ind in xrange(k):
        temp = np.matrix(.4 * np.random.randn(2, 2))
        covariances[:, :, ind] = temp * temp.transpose()
    # generate points
    points = generate_points(means, covariances, num_points)
    # classify using LDA
    lda_groups = lq_classify(points, means, covariances, priors, 0)
    # plot
    status = plot_classifs(lda_groups, points, line_specs, \
                           the_title='LDA results', filename='lda_qda_1.png')
    # classify using QDA
    qda_groups = lq_classify(points, means, covariances, priors, 1)
    # plot
    status = plot_classifs(qda_groups, points, line_specs, \
                           the_title='QDA results', filename='lda_qda_2.png')
    status = plot_classifs(actual_groups, points, line_specs, \
                           the_title='Actual', filename='lda_qda_3.png')