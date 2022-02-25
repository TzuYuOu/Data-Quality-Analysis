import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder # transform to 0/1/2
from scipy import stats #pointbiserialr
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model  import LogisticRegression
from sklearn import tree, metrics
from statsmodels.stats.weightstats import ztest
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy

from plotly.offline import plot
import plotly.graph_objs as go

class Accuracy:
    def __init__(self, data, sample_dict):
        self.data = data
        self.sample_dict = sample_dict
    
    def get_data(self):
        return self.data

    def coef(self):
        data = self.get_data()
        y_column_name = self.sample_dict['y']
        tpe = self.sample_dict[self.sample_dict['y']].split('_')[1]
        cate= self.sample_dict['problem']

        if len(data.dtypes[data.dtypes == 'object']) != 0:
            # data = self.change_to_dummy_variable()
            data = self.change_to_label_variable()
        
        y = data[[y_column_name]]
        x = data.drop(y_column_name, axis=1)
        # x = new_data.drop(y_column_name, axis=1)
        
        # self.corr_pearson = data.corr(method='pearson')
        # self.corr_kendall = data.corr(method='kendall')
        # self.corr_spearman = data.corr(method='spearman')
        save_result = {}
        save_beta = {}
        
        for i in range(len(x.columns)):
            one_x = x[x.columns[i]]
            temp = pd.concat([one_x, y], axis=1)
            Type = self.sample_dict[x.columns[i]].split('_')[1]
            if Type == 'ordinal' and cate=='regression': # x Continuous and y continuous
                correlation = temp.corr(method='pearson').iloc[1,0]
                
            elif Type == 'nominal' and tpe == 'ordinal': # x  not Continuous and y not continuous(ordinal)
                correlation = temp.corr(method='spearman').iloc[1,0]
            
            elif Type == 'nominal' and cate=='regression': # x not Continuous and y continuous
                correlation = stats.pointbiserialr(one_x, y)[0]
            
            elif Type == 'nominal' and cate=='classification': # x not Continuous and y not continuous(nominal)
                correlation = matthews_corrcoef(one_x, y) # phi-correlation
            
            else: # x Continuous and y not continuous
                correlation = stats.pointbiserialr(np.squeeze(y), one_x)[0]
            
            normalize_one_x = (one_x-one_x.min())/(one_x.max()-one_x.min())
            
            if cate=='classification':
                normalize_one_x = normalize_one_x.values.reshape(-1,1)
                lr=LogisticRegression(fit_intercept=True)
                lr.fit(normalize_one_x, y.values.ravel())
                
                coef = lr.coef_[0][0]
            else:
                model_ols = sm.OLS(y, normalize_one_x)
                results = model_ols.fit()
                coef = results.params.values[0]
            save_beta[x.columns[i]] = coef
            
            if np.sign(coef*correlation) == 1:
                Accuracy_Score = 1
            else:
                Accuracy_Score = 0
            
            save_result[x.columns[i]] = Accuracy_Score
        
        res = {}
        res['acc_res'] = save_result
        res['acc_beta'] = save_beta

        return res
    
    def change_to_label_variable(self):
        data = self.get_data()
        for i in range(len(data.columns)):
            if data.dtypes[i] == 'object':
                labelencoder = LabelEncoder()
                data[data.columns[i]] = labelencoder.fit_transform(data[data.columns[i]])

        return data

class InfoContent:
    def __init__(self, data, column_dict):
        self.data = data
        self.column_dict = column_dict
        self.y = self.column_dict['y']
        self.problem = self.column_dict['problem']
        
    def get_data(self):
        return self.data

    def cal_class_info(self, x, y):
        data = self.get_data()
        # save x's different values in unique_val
        unique_val = data[x].unique()
        ent = 0
        # calculate entropy
        for v in unique_val:
            tmp_df = data.loc[data[x] == v, y]
            cate = tmp_df.value_counts()
            ent += entropy([i/len(tmp_df) for i in cate], base=2) * len(tmp_df) / len(data) 
        return ent

    def cal_gain_ratio(self, x, y):
        data = self.get_data()
        y_list = list(pd.value_counts(data[y]))
        info = entropy(y_list, base=2)

        info_class = self.cal_class_info(x, y)

        gain_class = info - info_class

        x_list = list(pd.value_counts(data[x]))
        split_info = entropy(x_list, base=2)
        
        gain_ratio = round((gain_class / split_info), 4)

        return gain_ratio

    def cal_gain_ratio_score(self, x ,y):
        data = self.get_data()

        gain_ratio = self.cal_gain_ratio(x, y)

        # calculate upper bound
        # new_col = [1 for _ in range(len(data[x]))]
        # new_col[-1] = 0
        # data['New_Col'] = new_col
        # upper_bound = self.cal_gain_ratio('New_Col', y)

        # if((gain_ratio / upper_bound) >= 1):
        #     score = 1
        # else:
        #     score = 1-(gain_ratio/upper_bound)

        # return int(100 * score)
        return int(100 * gain_ratio)

    def get_gain_ratio(self):
        res = {}

        if self.problem == "classification":
            for col in self.data.columns:
                if(col != self.y):
                    res[col] = self.cal_gain_ratio_score(col, self.y)

        elif self.problem == "regression":
            return res
        else:
            return res
    
        # sort by value to descending order
        res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

        return res

    def get_variance_score(self):
        res = {}

        for col in self.data.columns:
            if(col != self.y and self.column_dict[col] == 'I_ordinal' or self.column_dict[col] == 'I_nominal'):
                samples_var = []
                # random sample 30 times
                for _ in range(30):
                    s = np.random.uniform(min(self.data[col]), max(self.data[col]), len(self.data))
                    samples_var.append(np.var(s))

                if np.mean(samples_var) <= 0:
                    var_ratio = 0
                else:
                    var_ratio = np.var(self.data[col]) / np.mean(samples_var)

                if var_ratio >= 1:
                    res[col] = 100
                else:
                    res[col] = int(100 * var_ratio)
        
        # sort by value to descending order
        res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    
        return res

class Completeness:
    def __init__(self, data, column_dict):
        self.data = data
        self.column_dict = column_dict
        self.null_score = 0
        self.mece_score = 0
    
    def get_data(self):
        return self.data
    
    def mece(self):
        
        data = self.get_data()
        y_column_name = self.column_dict['y']
        cate = self.column_dict['problem']
        
        if len(data.dtypes[data.dtypes == 'object']) != 0:
            data = self.change_to_dummy_variable()
            # data = self.change_to_label_variable()
        
        if cate != 'classification':
            
            y = data[[y_column_name]]
            x_columns = data.drop(y_column_name, axis=1)
            
            rs_h0 = []
            rs_h1 = []
            rs_history = pd.DataFrame(columns = ['R2' , 'random'])
            for i in range(30):
                model_h0 = sm.OLS(y, x_columns)
                results_h0 = model_h0.fit()
                rs_h0.append(results_h0.rsquared)
                x_columns['rand']= np.random.rand(len(y),1)
                
                model_h1 = sm.OLS(y,x_columns)
                results_h1 = model_h1.fit()
                rs_h1.append(results_h1.rsquared)
                rs_history.loc[i] = [results_h1.rsquared, x_columns['rand']]
                    
            z_test = ztest(rs_h0, rs_h1)
            rs_history = rs_history.sort_values(by=['R2'], ascending=False)[0:3]
            
            if z_test[1] <= 0.05:
                mean0 = sum(rs_h0) / len(rs_h0)
                mean1 = sum(rs_h1) / len(rs_h1)
                mece_score = mean0 / mean1
                return mece_score, rs_history.random.to_dict()
            else:
                mece_score = 1
                return mece_score, rs_history.random.to_dict()
        else:
            data = self.check_data_imbalanced(data, y_column_name)
            
            y = data[[y_column_name]]
            
            x_columns = data.drop(y_column_name, axis=1)
        
            acc_h0 = []
            acc_h1 = []
            # acc_history = pd.DataFrame(columns = ['acc' , 'random'])
            for i in range(30):
                model_h0 = tree.DecisionTreeClassifier()
                results_h0 = model_h0.fit(x_columns, y)
                acc_h0.append(metrics.accuracy_score(y, model_h0.predict(x_columns)))
                x_columns['rand']= np.random.rand(len(y),1)
                
                model_h1 = tree.DecisionTreeClassifier()
                results_h1 = model_h1.fit(x_columns, y)
                acc_h1.append(metrics.accuracy_score(y, model_h1.predict(x_columns)))
                # acc_history.loc[i] = [metrics.accuracy_score(y, model_h1.predict(x_columns)), x_columns['rand']]
                
        if acc_h0 == acc_h1:
            mece_score = 1
            # acc_history = acc_history.sort_values(by=['acc'], ascending=False)[0:3]
        else:
            z_test = ztest(acc_h0, acc_h1)
            # acc_history = acc_history.sort_values(by=['acc'], ascending=False)[0:3]
            
            if z_test[1] <= 0.05:
                mean0 = sum(acc_h0) / len(acc_h0)
                mean1 = sum(acc_h1) / len(acc_h1)
                mece_score = mean0 / mean1
            else:
                mece_score = 1
        return mece_score, {}
    
    def change_to_label_variable(self):
        data = self.get_data()
        for i in range(len(data.columns)):
            if data.dtypes[i] == 'object':
                labelencoder = LabelEncoder()
                data[data.columns[i]] = labelencoder.fit_transform(data[data.columns[i]])

        return data
    
    def change_to_dummy_variable(self):
        new_data = self.get_data()
        data = self.get_data()
        for i in range(len(data.columns)):
            if data.dtypes[i] == 'object':
                one_hot = pd.get_dummies(data[data.columns[i]])
                one_hot.columns = data.columns[i] + '_' + one_hot.columns
                
                # Drop column as it is now encoded
                new_data = new_data.drop(data.columns[i],axis = 1)
                    
                if len(one_hot.columns) < 20:
                    # Join the encoded df
                    new_data = new_data.join(one_hot)
        return new_data
    
    def check_data_imbalanced(self, data, y_column_name):
        index = data.groupby(y_column_name).count().iloc[:,0].max() / data.groupby(y_column_name).count().iloc[:,0].min()
        if index > 10:
            ## down_sampling refer from Bo-chen
            y = data[[y_column_name]]
            x_columns = data.drop(y_column_name, axis=1)
            
            temp = pd.DataFrame(data.groupby(y_column_name).count().iloc[:,0])
            temp['size'] = temp.iloc[:,0] / temp.iloc[:,0].min()
            temp['target_value'] = np.where(temp['size'] > 10, temp.iloc[:,0].min()*10, temp.iloc[:,0])
            num_class = temp['target_value'].to_dict()
            rus = RandomUnderSampler(sampling_strategy=num_class, random_state=0)
            down_X, down_y = rus.fit_resample(x_columns, y)
            new_data = pd.concat([down_X, down_y], axis=1)
        else:
            new_data = data
            
        return new_data
    
    def overview(self,data):
        self.overview_num = data.describe()
        self.overview_cat = data.describe(include=['O'])
        plt.style.use("seaborn")
        fig, axes = plt.subplots(len(self.overview_cat.columns),1)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(len(self.overview_cat.columns)):
            columns = self.overview_cat.columns
            axes[i].bar(data[columns[i]].unique(), data[columns[i]].value_counts(), width = 0.5)
            for j, k in enumerate(data[columns[i]].value_counts()):
                axes[i].text(j, k, str(k), fontweight="bold")
        
    def null_count(self, data):
        result_null = {}
        all_columns = data.columns
        all_null = 0
        for i in range(len(all_columns)):
            null_counts = data[all_columns[i]].isnull().sum()
            result_null[all_columns[i]] = null_counts
            all_null += null_counts
        self.null_score = 1 - (all_null / data.size)
        self.result = result_null
        return self.result

class Overview:
    def __init__(self, data, column_dict):
        self.data_old = data
        self.data = data
        self.column_dict = column_dict
    
    def missingvalue_pickup(self):
        self.column_withmiss = []
        self.column_missing_ratio = self.data.isnull().sum() / len(self.data)  # Column 遺漏值比例
        self.missvalue_index = []
        self.nonzero_missing_ratio = {}
        for key in self.column_missing_ratio.keys():
            if self.column_missing_ratio[key] != 0:
                self.column_withmiss.append(key)
                self.nonzero_missing_ratio[key] = self.column_missing_ratio[key]
            else:
                pass
                
        for key in self.data.keys():
            for i in self.data.index:
                if str(self.data[key][i]) == "nan":
                    self.missvalue_index.append(i)
                else:
                    pass
        
        self.data = self.data.drop(list(set(self.missvalue_index)))

    def column_parse(self):    ### 將資料欄位分類，確認為數值型或是類別型  Null need to be fill or separated before this operation
        self.C_nominal = []
        self.C_ordinal = []
        self.I_nominal = []
        self.I_ordinal = []
        self.other = []

        for key in self.column_dict.keys():
            if self.column_dict[key].lower() == 'c_nominal':
                self.C_nominal.append(key)
            
            elif self.column_dict[key].lower() == 'c_ordinal':
                self.C_ordinal.append(key) 
                                
            elif self.column_dict[key].lower() == 'i_nominal':
                self.I_nominal.append(key)
            
            elif self.column_dict[key].lower() == 'i_ordinal':
                self.I_ordinal.append(key)

            elif self.column_dict[key].lower() == 'other':
                self.other.append(key)
                

        if self.C_nominal:           
            self.data[self.C_nominal] = self.data[self.C_nominal].astype('category')
        
        if self.C_ordinal:
            self.data[self.C_ordinal] = self.data[self.C_ordinal].astype('category')
    
        if self.I_nominal:
            self.data[self.I_nominal] = self.data[self.I_nominal].astype('int64')

        if self.I_ordinal:
            self.data[self.I_ordinal] = self.data[self.I_ordinal].astype('int64')
          
    def overview_data(self):        
        self.overview = {   
            "Variables": int(len(self.data_old.columns)),
            "Observations": int(len(self.data_old)),
            "NumCategory": int(len(self.C_ordinal) + len(self.C_nominal)),
            "NumNumeric": int(len(self.I_ordinal) + len(self.I_nominal)),
            "C_nominal": self.C_nominal,
            "C_ordinal": self.C_ordinal,
            "I_nominal": self.I_nominal,
            "I_ordinal": self.I_ordinal,
            "Other": self.other,
            # "Variable with missing values": self.column_withmiss,
            "var_missing_ratio": self.nonzero_missing_ratio
        }

        return self.overview

    def overview_variable_cat(self):
        variable_cate = self.C_nominal + self.C_ordinal
        var_overview = self.data_old[variable_cate].describe()
        self.overview_category = {}
        for key in variable_cate:
            data_without_na = self.data_old[key].dropna()
            cate_name = []
            cate_ratio = []
            if(len(data_without_na.value_counts()) > 5):
                cate_name = list(data_without_na.value_counts().index[:5])
                cate_name.append('other values')
                
                cate_ratio = list(data_without_na.value_counts()[:5])
                cate_ratio.append(len(data_without_na) - sum(cate_ratio))
            else:
                cate_name = list(data_without_na.value_counts().index)
                cate_ratio = list(data_without_na.value_counts())

            cate_dict = {}
            for k, value in zip(cate_name, cate_ratio):
                cate_dict[k] = value
            
            fig = go.Bar(
                x=list(cate_dict.keys()), 
                y=list(cate_dict.values()),
            )
            graph = plot([fig], output_type='div')

            all_cate_ratio = list(data_without_na.value_counts() / sum(data_without_na.value_counts()))
            imbalance_ratio = max(all_cate_ratio) / min(all_cate_ratio)
            distinct_ratio = 100 * (var_overview[key]['unique']/var_overview[key]['count'])

            category_char = {
                "Missing_Ratio": 100 * self.column_missing_ratio[key],
                "Distinct": var_overview[key]['unique'],
                "Distinct_Ratio": distinct_ratio,
                "Imbalance_Ratio": imbalance_ratio,
                "Category_dict": cate_dict,
                "Graph": graph
            }
        
            self.overview_category.setdefault(key, category_char)
        
        return self.overview_category
        
    def overview_variable_num(self):
        variable_num = self.I_nominal + self.I_ordinal
        var_overview = self.data_old[variable_num].describe()
        self.overview_numeric = {}

        for key in variable_num:
            # draw histogram of the variable
            data_without_na = self.data_old[key].dropna()
            fig = go.Histogram(x=data_without_na)
            graph = plot([fig], output_type='div')

            numeric_char = {
                "Missing_Ratio": 100*self.column_missing_ratio[key],           
                "Mean": var_overview[key]['mean'],
                "Std": var_overview[key]['std'],
                "Min": var_overview[key]['min'],
                "1th_quantile": var_overview[key]['25%'],
                "Median": var_overview[key]['50%'],
                "3rd_quantile": var_overview[key]['75%'],
                "Max": var_overview[key]['max'],
                "Kurt": kurtosis(data_without_na),
                "Skew": skew(data_without_na),
                "Graph": graph
            }

            self.overview_numeric.setdefault(key, numeric_char)

        return self.overview_numeric

    def get_summary(self):
        # check missing value alarm
        missing_ratio_threshold = 0.1
        missing_ratio_alarm = {}
        for name in self.column_withmiss:
            if self.column_missing_ratio[name] > missing_ratio_threshold:
                missing_ratio_alarm[name] = self.column_missing_ratio[name]*100
        
        # check data imbalance
        variable_cate = self.C_nominal + self.C_ordinal + self.I_ordinal
        imbalance_cate = {}
        for key in variable_cate:
            data_without_na = self.data_old[key].dropna()
            all_cate_ratio = list(data_without_na.value_counts() / sum(data_without_na.value_counts()))
            imbalance_ratio = max(all_cate_ratio) / min(all_cate_ratio)
            # all_cate_ratio = list(self.data[key].value_counts() / sum(self.data[key].value_counts()))
            # print(all_cate_ratio)
            # imbalance_ratio = max(all_cate_ratio) / min(all_cate_ratio)
            if imbalance_ratio > 10:
                imbalance_cate.setdefault(key, imbalance_ratio)
        imbalance_cate = {k: v for k, v in sorted(imbalance_cate.items(), key=lambda item: item[1], reverse=True)}
       
        summary = {
            "missing_ratio_alarm": missing_ratio_alarm,
            'Imbalance_Alarm': imbalance_cate
        }

        return summary
