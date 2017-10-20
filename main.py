import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io as sio

# large plots for ipython notebook
# matplotlib.rcParams['figure.figsize'] = (17.0, 17.0)


class Estimator:
    def __init__(self, initial, landmarks, alpha=[0.5, 0.01, 0.01, 0.5], sigR=0.5, sigB=0.75, al=1.0 ,kap=1.0):
        self.x_hat = initial
        self.P = np.diag([1., 1., 0.5]) # np.eye(self.x_hat.shape[0]) * 0.1
        self.landmarks = landmarks
        self.alpha = alpha
        self.R = np.diag([sigR,sigB])

        self.lamb = al**2*(5 + kap) - 5
        self.gamma = np.sqrt(5 + self.lamb)
        self.w_m = np.concatenate(([[self.lamb/(5 + self.lamb)]], np.ones((1,2*5))/2/(5 + self.lamb)), axis=1)
        self.w_c = np.concatenate(([[self.lamb/(5 + self.lamb) + (1 - al**2 + 2)]], np.ones((1,2*5))/2/(5 + self.lamb)), axis=1)
        print self.w_c
        print sum(self.w_c[0,:])
        print sum(self.w_m[0,:])

    def propagate(self, dt, u):
        if u.sum() == 0:
            v = 0.000001
            w = 0.000001
        else:
            v = u[0,0]
            w = u[1,0]
        print "!!!!!propagate!!!!!"
        x_a = np.concatenate((self.x_hat, np.zeros((2, 1))), axis=0)
        Qu = np.diag([(self.alpha[0]*np.abs(v)**2 + self.alpha[1]*np.abs(w)**2), (self.alpha[2]*np.abs(v)**2 + self.alpha[3]*np.abs(w)**2)])
        Z_2 = np.zeros((2,2))
        Z_3 = np.zeros((3,2))
        P_a = np.asarray(np.bmat([[self.P, Z_3], [Z_3.T, Qu]]))

        L = np.linalg.cholesky(P_a)
        chi_a = np.concatenate((x_a, x_a + self.gamma*L, x_a - self.gamma*L), axis=1)

        # g(u + chi_u,chi_x)
        for i in range(2*5 + 1):
            chi_a[0:3,i:i+1] = self.dynamics(dt, chi_a[0:3,i:i+1], u + chi_a[3:5,i:i+1])

        self.x_hat = np.atleast_2d(np.sum(self.w_m*chi_a[0:3,:],axis=1)).T

        self.P = (self.w_c*(chi_a[0:3,:] - self.x_hat)).dot((chi_a[0:3,:] - self.x_hat).T)

        self.chi_a = chi_a

        return self.x_hat, self.P

    def update(self, z):
        print "-----update-----"
        # compute the difference betwee then predicted and measured 22 landmark measurements
        # (i.e range and bearing for 11 landmarks)
        # then remove all the nans, only leaving the landmarks we can see
        # additionally, visible_landmarks contains a vector of visible landmark indexes
        visible_landmarks = np.where(~np.isnan(z[::2, 0]))[0]

        for vis_lm in visible_landmarks:

            # regenerate simga points
            x_a = np.concatenate((self.x_hat, np.zeros((2, 1))), axis=0)
            Z_3 = np.zeros((3,2))
            P_a = np.asarray(np.bmat([[self.P, Z_3], [Z_3.T, self.R]]))

            L = np.linalg.cholesky(P_a)
            self.chi_a = np.concatenate((x_a, x_a + self.gamma*L, x_a - self.gamma*L), axis=1)

            meas_idx = np.array([2*vis_lm, 2*vis_lm])

            Zbar = np.empty((2,2*5 + 1))
            for j in range(2*5 + 1):
                Zbar[:,j:j+1] = self.measure(self.chi_a[0:3,j:j+1])[meas_idx] + self.chi_a[3:5,j:j+1]

            zhat = np.atleast_2d(np.sum(self.w_m*Zbar,axis=1)).T

            S = (self.w_c*(Zbar - zhat)).dot((Zbar - zhat).T)
            P_Ct = (self.w_c*(self.chi_a[0:3,:] - self.x_hat)).dot((Zbar - zhat).T)
            K = P_Ct.dot(np.linalg.inv(S))

            residual = z[meas_idx] - zhat
            if residual[1,:] > np.pi:
                residual[1,:] -= 2*np.pi
            if residual[1,:] < -np.pi:
                residual[1,:] += 2*np.pi
            print "Residual", residual

            dist = residual.T.dot(np.linalg.inv(S)).dot(residual)[0,0]
            if dist < 9:
                self.x_hat = self.x_hat + K.dot(residual)
                self.P = self.P - K.dot(S).dot(K.T)
            else:
                print "gated a measurement", np.sqrt(dist)

        return self.x_hat, self.P

    def measure(self, x, Q=None):
        # prevent anyone from making a mistake here
        assert x.shape == (3, 1) and (
        Q is None or Q.shape == (3, 3)), "bad shapes for measurement: {} should be (3,1) and {} should be (3,3)".format(
            u.shape, x.shape)

        z = np.zeros([2 * self.landmarks.shape[0]])

        for i, m in enumerate(self.landmarks):
            q = (m[0] - x[0]) ** 2 + (m[1] - x[1]) ** 2
            z[i * 2] = np.sqrt(q)
            z[i * 2 + 1] = np.arctan2(m[1] - x[1], m[0] - x[0]) - x[2]

        z = np.random.multivariate_normal(z, Q) if Q else z

        return z[:, None]

    def dynamics(self, dt, x, u):
        # prevent anyone from making a mistake here
        assert u.shape == (2, 1) and x.shape == (3, 1), "bad shapes for dynamics: {} should be (2,1) and {} should be (3,1)".format(u.shape, x.shape)

        noise_v, noise_omega = 0, 0
        v, omega = u[0] + noise_v, u[1] + noise_omega

        if omega == 0:
            omega = 0.00000001

        px, py, theta = x[0], x[1], x[2]
        px += v / omega * np.sin(theta + omega * dt) - v / omega * np.sin(theta)
        py += v / omega * np.cos(theta) - v / omega * np.cos(theta + omega * dt)
        theta += omega * dt

        return np.array([px, py, theta])


matfile = sio.loadmat('processed_data.mat')
landmark_bearing = matfile['l_bearing'].T
landmark_range = matfile['l_depth'].T
landmark_t = matfile['l_time'][:, 0]
landmarks = matfile['landmarks']
odometry_t = matfile['odom_t'][0]
odometry_vel = matfile['vel_odom'].T
odometry_pos = matfile['pos_odom_se2'].T
x_hat = odometry_pos[0, None].T  # (3, 1)

# Fix times so we start at 0
odometry_t = odometry_t - np.min(odometry_t)
landmark_t = landmark_t - np.min(odometry_t)

estimator = Estimator(initial=x_hat, landmarks=landmarks)
measurement_index = 0
x_history = []
p_history = []

plt.figure(0)
plt.ion()

for i, (t, dt) in enumerate(zip(odometry_t, np.diff(np.concatenate([[0],odometry_t])))):
    print "--------------------------"
    print "i =", i
    u = odometry_vel[i, None].T  # (2, 1)

    # if we have stepped over a landmark measurement, and there exists a measurement
    if measurement_index < landmark_t.shape[0] and t > landmark_t[measurement_index]:
        # compute the time since the last propagate and this measurement update
        fractional_dt = landmark_t[measurement_index] - (t - dt)

        # build the measurement, [b_0, r_0, b_1, r_1, ... , b_n, r_n]
        z = np.ravel(np.column_stack([landmark_range[measurement_index],
                                      landmark_bearing[measurement_index]]))[:, None]

        # this propagate step will be the last odometry to the landmark sensor measurement
        # x_hat, P = estimator.propagate(fractional_dt, u)

        # update the model
        x_hat, P = estimator.update(z)

        # the next propagate step will be the time from after the measurement
        # to the next odometry message
        # dt = t - landmark_t[measurement_index]

        measurement_index += 1

    x_hat, P = estimator.propagate(dt, u)
    # print np.linalg.eigvals(P)

    x_history.append(x_hat[:, 0])
    p_history.append(P)

    [[x],[y],[t]] = x_hat
    if False:
        plt.clf()
        plt.axis([-3, 9, -3, 15])
        plt.scatter(x, y)
        plt.scatter(x + 0.25*np.cos(t), y + 0.25*np.sin(t), color='r')
        plt.plot(landmarks.T[0], landmarks.T[1], 'o', label='landmarks')
        plt.pause(0.5)
plt.ioff()

x_history = np.array(x_history)
p_history = np.array(p_history)

# plot the x,y,theta
# and confidence intervals
# plt.subplot(2, 1, 1)
plt.figure(1)
[plt.plot(odometry_t, y.T, label=l) for y, l in zip(odometry_pos.T, ['x_odometry', 'y_odometry', 'theta_odometry'])]
[plt.plot(odometry_t, y.T, label=l) for y, l in zip(x_history.T, ['x_hat', 'y_hat', 'theta_hat'])]
[plt.plot(odometry_t, x_history[:, i] + 3.0 * p_history[:, i, i] ** 0.5, '-k', alpha=0.5) for i in range(3)]
[plt.plot(odometry_t, x_history[:, i] - 3.0 * p_history[:, i, i] ** 0.5, '-k', alpha=0.5) for i in range(3)]
plt.legend()

# plot the global position and landmarks
# plt.subplot(2, 1, 2)
plt.figure(2)
plt.plot(odometry_pos.T[0], odometry_pos.T[1], label='odometry_position')
plt.plot(x_history.T[0], x_history.T[1], label='estimated_position')
plt.plot(landmarks.T[0], landmarks.T[1], 'o', label='landmarks')
# plt.legend(loc='upper left')

plt.figure(3)
plt.plot(odometry_t, x_history[:,2])
plt.show()

plt.figure(3)
