import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

real_samp, _ = generate_real_sample(100)
fake_samp, _ = generate_fake_sample(generator, 5, 100)

psd_real = []
psd_fake = []
for i in range(100):
  vals_real, _ = plt.psd(real_samp[i])
  vals_fake, _ = plt.psd(fake_samp[i])
  psd_real.append(10*np.log(vals_real))
  psd_fake.append(10*np.log(vals_fake))
psd_real = np.array(psd_real)
psd_fake = np.array(psd_fake)
plt.show()
print(psd_fake.shape)
means_real  = []
means_fake = []
std_real = []
std_fake = []
for i in range(129):
  means_real.append(np.mean(psd_real[:,i]))
  means_fake.append(np.mean(psd_fake[:,i]))
  std_real.append(np.std(psd_real[:,i]))
  std_fake.append(np.std(psd_fake[:,i]))
  
fs = np.linspace(0,128,129)
means_real = np.array(means_real)
means_fake = np.array(means_fake)
std_real = np.array(std_real)
std_fake = np.array(std_fake)
plt.subplot(221)
plt.plot(fs, means_real, color = 'orange')
y1 = means_real + std_real
y2 = means_real - std_real
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
#plt.yscale('log')
plt.title('real PSD plot')
plt.subplot(222)
plt.plot(fs, means_fake, color = 'orange')
y1 = means_fake + std_fake
y2 = means_fake - std_fake
plt.fill_between(fs, y2, y1, alpha = 0.2)
plt.grid()
#plt.yscale('log')
plt.title('fake PSD plot')
plt.subplot(223)
plt.plot(fs, std_real)
plt.title('real std')
plt.subplot(224)
plt.plot(fs, std_fake)
plt.title('fake std')
plt.tight_layout()
