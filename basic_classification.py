import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class BasicClassification:
    
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.rf_model = None
        self.X = None
        self.y = None
    
    def display_initial_data(self):
       
        print(self.df.head())
        print("Mevcut Başlıklar:", self.df.columns)
    
    def clean_data(self):
       
        self.df.columns = ['Product', 'Sales_volume(kg)', 'Brand', 'Price(TL)',
                           'Price(TL/Sales_volume)', 'Quality_Score', 'Delivery_Time', 'Fav_number',
                           'Comment_Count', 'Store']

        
        self.df['Brand'] = self.df.groupby(['Product', 'Price(TL/Sales_volume)'])['Brand'].transform(lambda x: x.ffill().bfill())

        self.df['Comment_Count'] = self.df['Comment_Count'].fillna(self.df['Comment_Count'].mean())
        self.df['Price(TL)'] = self.df['Price(TL)'].str.replace(',', '').astype(float)
        self.df['Price(TL)'] = self.df['Price(TL)'].round(2)
        self.df['Price(TL/Sales_volume)'] = self.df['Price(TL/Sales_volume)'].round(2)

        self.df['Product'] = self.df['Product'].str.strip()
        self.df['Product'] = self.df['Product'].str.replace(' +', ' ', regex=True)
        self.df['Product'] = self.df['Product'].str.lower()

    def save_cleaned_data(self, file_name):
        self.df.to_csv(file_name, index=False)
    
    def detect_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    def remove_outliers(self, column):
        outliers, lower_bound, upper_bound = self.detect_outliers(column)
        self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
        self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])
        #print(f"Aykırı değerler {column} sütununda temizlendi.")
    
    def analyze_data(self):
        
        grouped_df = self.df.groupby('Product').agg(
            {
                'Sales_volume(kg)': ['min', 'max', 'mean', 'median', 'std'],
                'Price(TL)': ['min', 'max', 'mean', 'median', 'std'],
                'Price(TL/Sales_volume)': ['min', 'max', 'mean', 'median', 'std'],
                'Quality_Score': ['min', 'max', 'mean', 'median', 'std'],
                'Delivery_Time': ['min', 'max', 'mean', 'median', 'std'],
                'Fav_number': ['min', 'max', 'mean', 'median', 'std'],
                'Comment_Count': ['min', 'max', 'mean', 'median', 'std'],
                'Brand': lambda x: ', '.join(x.unique()),  
                'Store': lambda x: ', '.join(x.unique())
            }
        )
        grouped_df = grouped_df.reset_index()
        grouped_df = grouped_df.round(2)

        print(grouped_df)
        grouped_df.to_csv('grouped_market_data.csv')

    def visualize_correlation_matrix(self):
        numeric_cols = ['Sales_volume(kg)', 'Price(TL)', 'Price(TL/Sales_volume)', 'Quality_Score',
                        'Delivery_Time', 'Fav_number', 'Comment_Count']
        corr_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korelasyon Matrisi')
        plt.show()
    def visualize_strong_correlations(self, corr_matrix, threshold=0.7):
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                if col != idx:  # Aynı sütun ve satırı karşılaştırma
                    corr_value = corr_matrix.loc[idx, col]
                    if abs(corr_value) >= threshold:  # Güçlü ilişkileri seç
                        # Scatter plot ile ilişkili olanları görselleştir
                        plt.figure(figsize=(6, 4))
                        sns.scatterplot(x=self.df[col], y=self.df[idx])
                        plt.title(f"{col} ile {idx} ilişkisi (korelasyon: {corr_value:.2f})")
                        plt.xlabel(col)
                        plt.ylabel(idx)
                        plt.show()
    def interpret_correlation(self, corr_matrix, threshold=0.7):
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                if col != idx:
                    corr_value = corr_matrix.loc[idx, col]
                    if corr_value >= threshold:
                        print(f"{col} ile {idx} arasında güçlü bir pozitif ilişki var (korelasyon: {corr_value:.2f})")
                    elif corr_value <= -threshold:
                        print(f"{col} ile {idx} arasında güçlü bir negatif ilişki var (korelasyon: {corr_value:.2f})")
                    elif -threshold < corr_value < threshold:
                        print(f"{col} ile {idx} arasında anlamlı bir ilişki yok (korelasyon: {corr_value:.2f})")

    def train_random_forest(self):
       
        self.df['Category'] = self.df['Product'].str.lower().str.strip()
        self.df = pd.get_dummies(self.df, columns=['Category'], drop_first=True)

       
        self.X = self.df[['Quality_Score', 'Delivery_Time', 'Fav_number', 'Comment_Count', 'Price(TL/Sales_volume)', 'Sales_volume(kg)'] + list(self.df.columns[self.df.columns.str.startswith('Category_')])]
        self.y = self.df['Price(TL)']

    
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

       
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)

    
        y_pred_rf = self.rf_model.predict(X_test)

        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = mse_rf ** 0.5
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        # Performans sonuçları:
        print(f"Random Forest Performans Metrikleri:")
        print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
        print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
        print(f"R-squared (R²) Skoru: {r2_rf:.2f}")

    def recommend_products(self):
        while True:
            product_list = self.df['Product'].unique()
            product_list_text = ', '.join(product_list)

            selected_product = input(f"Almak istediğiniz ürünü seçin (Mevcut ürünler: {product_list_text}) : ").lower().strip()

            if selected_product == 'q':
                print("Programdan çıkış yapılıyor.")
                break

            matching_products = [product for product in product_list if product.lower() == selected_product]

            if not matching_products:
                print(f"Seçtiğiniz ürün listemizde bulunmamaktadır, lütfen aşağıdaki ürünlerden birini seçin: {product_list_text}")
            else:
                selected_product = matching_products[0]
                print(f"Seçilen ürün: {selected_product}")

                user_budget = float(input("Bütçenizi girin: "))

                quality_score = float(input("Kalite puanını girin (1: En düşük, 2: Düşük, 3:Orta, 4:İyi, 5: Çok iyi): "))
                delivery_time = float(input("Teslimat puanı girin (1: En yavaş(2,5-6 saat), 2: Yavaş(max:2 saat), 3:Orta(max:60dk), 4:Hızlı(max.45dk), 5: Çok hızlı(max.30dk)): "))
                fav_number = int(input("Fav yapılma sayısını girin (0-...): "))
                comment_count = float(input("Yorum sayısını girin (0-...): "))

                user_input = {
                    'Quality_Score': [quality_score],
                    'Delivery_Time': [delivery_time],
                    'Fav_number': [fav_number],
                    'Comment_Count': [comment_count],
                    'Price(TL/Sales_volume)': [0],  
                    'Sales_volume(kg)': [0],  
                }

                for category in self.df.columns[self.df.columns.str.startswith('Category_')]:
                    user_input[category] = [1 if category == f'Category_{selected_product}' else 0]

                user_input_df = pd.DataFrame(user_input)

                predicted_price = self.rf_model.predict(user_input_df)

                print(f"Bu özelliklerle tahmin edilen fiyat: {predicted_price[0]:.2f} TL")

                filtered_df = self.df.loc[(self.df['Product'].str.lower().str.strip() == selected_product.lower()) & (self.df['Price(TL)'] <= user_budget)].copy()

                if filtered_df.empty:
                    print("Bütçenize uygun ürün bulunamadı.")
                else:
                    filtered_df.loc[:, 'Predicted_Price'] = self.rf_model.predict(filtered_df[self.X.columns])
                    filtered_df.loc[:, 'Price_Difference'] = abs(filtered_df['Price(TL)'] - predicted_price)

                    sorted_df = filtered_df.sort_values(by='Price_Difference', ascending=True)
                    suggestions = sorted_df[['Product', 'Sales_volume(kg)', 'Brand', 'Price(TL)', 'Predicted_Price', 'Price_Difference', 'Quality_Score', 'Delivery_Time', 'Store']].head(5)

                    print("Tahmini fiyata en yakın önerilen ürünler ve hangi mağazadan alınacağı:")
                    print(suggestions)

            print("\nBaşka bir ürün aramak için devam edebilirsiniz, çıkış yapmak için 'q' tuşuna basınız.\n")



file_path = 'market.csv'
classification = BasicClassification(file_path)
classification.display_initial_data()
classification.clean_data()
classification.save_cleaned_data('market_cleaned.csv')

classification.detect_outliers('Price(TL/Sales_volume)')

classification.remove_outliers('Price(TL/Sales_volume)')
classification.analyze_data()
classification.visualize_correlation_matrix()

corr_matrix = classification.df[['Sales_volume(kg)', 'Price(TL)', 'Price(TL/Sales_volume)', 'Quality_Score', 'Delivery_Time', 'Fav_number', 'Comment_Count']].corr()
classification.interpret_correlation(corr_matrix)
classification.visualize_strong_correlations(corr_matrix)

classification.train_random_forest()
classification.recommend_products()

