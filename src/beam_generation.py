import os
import numpy as np
import torch


class BeamGenerator:
    def __init__(self, windowsize=1024, resolution_um=0.1):
        self.windowsize = windowsize
        self.resolution = resolution_um
        self.x_um = np.arange(windowsize) * resolution_um
        self.center = windowsize * resolution_um / 2

    def _normalize_to_charge(self, y, charge):
        area = np.trapezoid(y, self.x_um)
        if area <= 0:
            raise ValueError("Profile area must be positive.")
        return y * (charge / area)

    def _gaussian(self, charge, sigma, center):
        y = np.exp(-(self.x_um - center) ** 2 / (2 * sigma ** 2))
        return self._normalize_to_charge(y, charge)

    def _asymmetric_gaussian(self, charge, sigma_left, sigma_right, center):
        x = self.x_um
        sigma = np.where(x < center, sigma_left, sigma_right)
        y = np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        return self._normalize_to_charge(y, charge)

    def _super_gaussian_flat(self, charge, sigma, center, order=4):
        x = self.x_um
        y = np.exp(-np.abs((x - center) / sigma) ** (2 * order))
        return self._normalize_to_charge(y, charge)

    def _super_gaussian_sharp(self, charge, sigma, center, order=0.7):
        x = self.x_um
        y = np.exp(-np.abs((x - center) / sigma) ** (2 * order))
        return self._normalize_to_charge(y, charge)

    def _generate_one(self, charge, sigma, center, shape_type):
        if shape_type == "gaussian":
            return self._gaussian(charge, sigma, center)

        elif shape_type == "asymmetric":
            ratio = np.random.uniform(0.25, 2.5)
            sigma_left = sigma
            sigma_right = sigma * ratio
            return self._asymmetric_gaussian(
                charge, sigma_left, sigma_right, center
            )

        elif shape_type == "super_flat":
            order = np.random.uniform(3, 8)
            return self._super_gaussian_flat(
                charge, sigma, center, order
            )

        elif shape_type == "super_sharp":
            order = np.random.uniform(0.5, 0.9)
            return self._super_gaussian_sharp(
                charge, sigma, center, order
            )

        else:
            raise ValueError("Unknown shape type.")

    def generate(self, charges, sigmas, distance=None, shape_types=None):
        """
        charges      : list
        sigmas       : list
        distance     :
            n=2  -> scalar d12
            n=3  -> tuple (d12, d23)
        shape_types  :
            list of same length as charges
            or None (random selection)
        """
        n = len(charges)
        y = np.zeros_like(self.x_um)

        if shape_types is None:
            shape_types = np.random.choice(
                ["gaussian", "asymmetric", "super_flat", "super_sharp"],
                size=n
            )

        if n == 1:
            c1 = self.center
            y += self._generate_one(
                charges[0], sigmas[0], c1, shape_types[0]
            )

        elif n == 2:
            if distance is None:
                raise ValueError("For 2 bunches, distance required.")

            d12 = distance
            c1 = self.center - d12 / 2
            c2 = self.center + d12 / 2

            y += self._generate_one(
                charges[0], sigmas[0], c1, shape_types[0]
            )
            y += self._generate_one(
                charges[1], sigmas[1], c2, shape_types[1]
            )

        elif n == 3:
            if distance is None or len(distance) != 2:
                raise ValueError("For 3 bunches, distance must be (d12, d23).")

            d12, d23 = distance
            total_span = d12 + d23

            c1 = self.center - total_span / 2
            c2 = c1 + d12
            c3 = c2 + d23

            y += self._generate_one(
                charges[0], sigmas[0], c1, shape_types[0]
            )
            y += self._generate_one(
                charges[1], sigmas[1], c2, shape_types[1]
            )
            y += self._generate_one(
                charges[2], sigmas[2], c3, shape_types[2]
            )

        else:
            raise ValueError("Supports only 1–3 bunches.")

        return self.x_um, y


def log_uniform(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def generate_dataset(
    n_samples,
    n_bunch,
    range_charge,
    range_sigma,
    range_distance=None,
    ratio_range_charge=None,
    save_dir="./dataset",
    windowsize=256,
    resolution=0.15,
    noise_level=0.0,
    noise_mode="relative_to_max",
    shape_prob=None,
):
    """
    noise_level:
        0.0  -> no noise
        0.01 -> 1% Gaussian noise

    shape_prob:
        dict like:
        {
            "gaussian": 0.3,
            "asymmetric": 0.3,
            "super_flat": 0.2,
            "super_sharp": 0.2
        }
        If None -> uniform random
    """
    os.makedirs(save_dir, exist_ok=True)

    generator = BeamGenerator(windowsize, resolution)
    shape_types_all = ["gaussian", "asymmetric", "super_flat", "super_sharp"]

    if shape_prob is not None:
        probs = [shape_prob[k] for k in shape_types_all]
    else:
        probs = None

    for i in range(n_samples):
        charges = []
        sigmas = []

        q1 = np.random.uniform(*range_charge)
        charges.append(q1)

        if n_bunch >= 2:
            r = log_uniform(*ratio_range_charge)
            q2 = q1 / r
            charges.append(q2)

        if n_bunch == 3:
            r2 = log_uniform(2, 5)
            q3 = q1 / r2
            charges.append(q3)

        for _ in range(n_bunch):
            sigmas.append(log_uniform(*range_sigma))

        distance = None
        if n_bunch == 2:
            distance = log_uniform(*range_distance)
        elif n_bunch == 3:
            d12 = log_uniform(*range_distance)
            d23 = log_uniform(*range_distance)
            distance = (d12, d23)

        shape_types = list(
            np.random.choice(shape_types_all, size=n_bunch, p=probs)
        )

        x, y = generator.generate(
            charges,
            sigmas,
            distance,
            shape_types=shape_types
        )

        if noise_level and noise_level > 0.0:
            y = np.asarray(y, dtype=np.float64)

            if noise_mode == "relative_to_value":
                sigma = noise_level * np.maximum(y, 0.0)
            else:
                y_max = float(np.max(y)) if np.max(y) > 0 else 1.0
                sigma = noise_level * y_max

            noise = np.random.randn(*y.shape) * sigma
            y = y + noise
            y = np.clip(y, 0.0, None)

        sample = {
            "x_um": torch.tensor(x, dtype=torch.float32),
            "Intensity": torch.tensor(y, dtype=torch.float32),
            "charges": [float(np.round(q, 5)) for q in charges],
            "sigmas": [float(np.round(s, 5)) for s in sigmas],
            "shapes": shape_types,
        }

        if n_bunch == 2:
            sample["distance_12"] = float(np.round(distance, 5))
        elif n_bunch == 3:
            sample["distance_12"] = float(np.round(distance[0], 5))
            sample["distance_23"] = float(np.round(distance[1], 5))

        shape_tag = "_".join(shape_types)

        if noise_level and noise_level > 0.0:
            noise_str = f"{noise_level:.4f}".rstrip("0").rstrip(".")
            filename = f"G{n_bunch}_{shape_tag}_sample_{i+1}_noise_{noise_str}.pt"
        else:
            filename = f"G{n_bunch}_{shape_tag}_sample_{i+1}.pt"

        torch.save(sample, os.path.join(save_dir, filename))

    print("Dataset generation completed.")
