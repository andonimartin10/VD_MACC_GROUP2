import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import metrics
import plotly.graph_objs as go
import colorlover as cl
from sklearn.svm import SVC, NuSVC
from sklearn.inspection import permutation_importance

df = pd.read_csv('data/heart.csv')
discrete_columns = []
continuous_columns = []

for col in df.columns:
    if col != 'HeartDisease':
        if df[col].dtype == 'object':
            discrete_columns.append(col)
        else:
            continuous_columns.append(col)
data_continuous = df[continuous_columns]
dummy_sex = pd.get_dummies(df['Sex'], prefix='sex')
dummy_CPT = pd.get_dummies(df['ChestPainType'], prefix='chest_p_t')
dummy_RECG = pd.get_dummies(df['RestingECG'], prefix='rest_ecg')
dummy_angina = pd.get_dummies(df['ExerciseAngina'], prefix='angina')
dummy_STslope = pd.get_dummies(df['ST_Slope'], prefix='st_slope')
df_v2 = pd.concat(
    [dummy_sex, dummy_CPT, dummy_RECG, dummy_angina, dummy_STslope, data_continuous, df['HeartDisease']], axis=1)
finaldf = df_v2.drop(['sex_F', 'chest_p_t_ASY', 'rest_ecg_LVH', 'angina_N', 'st_slope_Down'], axis=1)


class Dashboard(object):
    def __init__(self):
        self.df = finaldf
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0, stratify=self.y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None
        self.explainer = None

    def update_model(self, algorithms, c, nu, kernel):
        if algorithms == 'SVC':
            algorithm = SVC(C=c, kernel=kernel)
            self.model = algorithm.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
        elif algorithms == 'NuSVC':
            algorithm = NuSVC(nu=nu, kernel=kernel)
            self.model = algorithm.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

    def most_importance_features(self):
        results = permutation_importance(self.model, self.X, self.y, scoring='accuracy')
        importance = results.importances_mean

        fig = go.Bar(
            x=importance,
            y=self.df.columns,
            orientation='h')

        layout = go.Layout(
            title='Features Importance Mean',
            legend=dict(x=0, y=1.05, orientation="h"),
            margin=dict(l=50, r=10, t=55, b=40),
        )

        data = [fig]

        figure = go.Figure(data=data, layout=layout)

        return figure

    def serve_roc_curve(self):
        decision_test = self.y_pred
        fpr, tpr, threshold = metrics.roc_curve(self.y_test, decision_test)

        # AUC Score
        auc_score = metrics.roc_auc_score(y_true=self.y_test, y_score=decision_test)

        trace0 = go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name='Test Data',
        )

        layout = go.Layout(
            title=f'ROC Curve (AUC = {auc_score:.4f})',
            xaxis=dict(
                title='False Positive Rate'
            ),
            yaxis=dict(
                title='True Positive Rate'
            ),
            legend=dict(x=0, y=1.05, orientation="h"),
            margin=dict(l=50, r=10, t=55, b=40),
        )

        data = [trace0]
        figure = go.Figure(data=data, layout=layout)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return figure

    def serve_pie_confusion_matrix(self):
        matrix = metrics.confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)
        tn, fp, fn, tp = matrix.ravel()

        values = [tp, fn, fp, tn]
        label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
        labels = ["TP", "FN", "FP", "TN"]
        blue = cl.flipper()["seq"]["9"]["Blues"]
        red = cl.flipper()["seq"]["9"]["Reds"]
        colors = ["#11c6e9", blue[1], "#ff816d", "#ff944c"]
        f1score = f1_score(self.y_test, self.y_pred)

        trace0 = go.Pie(
            labels=label_text,
            values=values,
            hoverinfo="label+value+percent",
            textinfo="text+value",
            text=labels,
            sort=False,
            marker=dict(colors=colors),
            insidetextfont={"color": "white"},
            rotation=90,
        )

        layout = go.Layout(
            title=f"Confusion Matrix (F1-Score ={f1score: .4f})",
            margin=dict(l=50, r=10, t=55, b=40),
            legend=dict(font={"color": "#a5b1cd"}, orientation="h"),
            plot_bgcolor="#282b38",
            font={"color": "#282b38"},
        )

        data = [trace0]
        figure = go.Figure(data=data, layout=layout)

        return figure

    def get_indicators(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1score = f1_score(self.y_test, self.y_pred)
        rocauc = roc_auc_score(self.y_test, self.y_pred)
        return accuracy, f1score, rocauc

    def get_variable_names(self):
        variables = []
        for col in self.X_test.columns:
            var = {
                'label': col,
                'value': col
            }
            variables.append(var)
        return variables

    def get_columns(self, column_names):
        columns = self.df[column_names]
        return columns
