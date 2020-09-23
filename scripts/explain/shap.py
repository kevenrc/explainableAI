import shap

class Shap:
    def __init__(self, model_type, model, background=None, category=None):
        shap.initjs()
        self.model_type = model_type
        self.explainer = shap.KernelExplainer(model)
        self.category = category
        self.shap_values = None
        self.expected_value = None

    def explain(self, data):
        self.shap_values = self.explainer.shap_values(data)
        self.expected_value = self.explainer.expected_value

    def plot_summary(self, data):
        shap.summary_plot(self.shap_values, data, plot_type="bar")

    def plot_force(self, data, index):
        if self.expected_value is None:
            return
        if self.shap_values is None:
            return
        shap.force_plot(self.expected_value, self.shap_values[index, :],
                        data, link='logit', matplotlib=True)

    def plot_image(self, data, index):
        if self.expected_value is None:
            return
        if self.shap_values is None:
            return
        shap.image_plot(self.shap_values[index, :], data)

    def get_shap_values(self):
        return self.shap_values

    def get_expected_value(self):
        return self.expected_value
