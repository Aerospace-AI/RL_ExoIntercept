import numpy as np

class Adv_normalizer(object):
    @staticmethod
    def apply(u_adv, std_adv, advantages, scale_factor=1):
        advantages = (advantages - u_adv) / (scale_factor*std_adv)
        return advantages

    @staticmethod
    def apply_std_only(u_adv, std_adv, advantages, scale_factor=1):
        advantages = (advantages) / (scale_factor*std_adv)
        return advantages

    @staticmethod
    def apply_none(u_adv, std_adv, advantages, scale_factor=1):
        return advantages


class Adv_exp(object):
    def __init__(self, normalizer=Adv_normalizer.apply, adv_temp=1.0, clip=1.0, scale_factor=1.0):
        self.normalizer = normalizer
        self.adv_temp = adv_temp
        self.clip = clip
        self.scale_factor = scale_factor

    def calc_adv(self, advantages_unp, advantages):
        u_adv = advantages_unp.mean()
        std_adv = advantages_unp.std() +  1e-6

        advantages = self.normalizer(u_adv, std_adv, advantages, scale_factor=self.scale_factor)

        advantages = np.exp(advantages / self.adv_temp)
        advantages = np.clip(advantages, 0.0, self.clip)
        return advantages

class Adv_exp_clipfirst(object):
    def __init__(self, normalizer=Adv_normalizer.apply, adv_temp=np.e, clip=3.0,  scale_factor=1.0):
        self.normalizer = normalizer
        self.adv_temp = adv_temp
        self.clip = clip
        self.scale_factor = scale_factor

    def calc_adv(self, advantages_unp, advantages):
        u_adv = advantages_unp.mean()
        std_adv = advantages_unp.std() +  1e-6

        advantages = self.normalizer(u_adv, std_adv, advantages, scale_factor=self.scale_factor)
        advantages = np.clip(advantages, -self.clip, self.clip)
        advantages = np.exp(advantages / self.adv_temp)
        return advantages

class Adv_relu(object):
    def __init__(self, normalizer=Adv_normalizer.apply, clip=3.0, scale_factor=1.0):
        self.normalizer = normalizer
        self.clip = clip
        self.scale_factor = scale_factor

    def calc_adv(self, advantages_unp, advantages):
        u_adv = advantages_unp.mean()
        std_adv = advantages_unp.std() +  1e-6

        advantages = self.normalizer(u_adv, std_adv, advantages, scale_factor=self.scale_factor)
        advantages = np.clip(advantages, 0.0, self.clip)

        return advantages

class Adv_relu2(object):
    def __init__(self, normalizer=Adv_normalizer.apply, clip=1.0):
        self.normalizer = normalizer
        self.clip = clip

    def calc_adv(self, advantages_unp, advantages):

        idx = np.where(advantages > 0)[0]
        new_advantages = np.zeros_like(advantages)
        new_advantages[idx] = 1.

        return new_advantages

class Adv_default(object):
    def __init__(self, normalizer=Adv_normalizer.apply, clip=3.0,  scale_factor=1.0):
        self.normalizer = normalizer
        self.clip = clip
        self.scale_factor = scale_factor

    def calc_adv(self, advantages_unp, advantages):
        u_adv = advantages_unp.mean()
        std_adv = advantages_unp.std() +  1e-6

        advantages = self.normalizer(u_adv, std_adv, advantages,  scale_factor=self.scale_factor)
        advantages = np.clip(advantages, -self.clip, self.clip)

        return advantages

