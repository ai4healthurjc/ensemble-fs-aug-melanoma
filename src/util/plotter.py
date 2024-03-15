import matplotlib.pyplot as plt


def plot_feature_drop_ratio(df_results):
    print('feature drop ratio')


def bootstrap_graphic(bootstrap_df, filepath, size1=5.2, size2=8.6):
    fig, ax = plt.subplots(figsize=(size1, size2))

    colors = []
    labels = []

    for index, row in bootstrap_df.iterrows():
        var_name = row['var_name']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        ci_center = (ci_lower + ci_upper) / 2

        if ci_center < 0:
            color = 'blue'
            label = 'Not melanoma lesions'
        else:
            color = 'green'
            label = 'Melanoma lesions'

        colors.append(color)
        labels.append(label)

        # Dibujar una línea de puntos que llegue hasta el final de la barra
        ax.plot([ci_lower, ci_upper], [var_name, var_name], color=color, linestyle='-', zorder=4)

        # Dibujar el punto en el centro del intervalo
        ax.scatter(ci_center, var_name, color=color, marker='o', zorder=3, s=15)

        # Dibujar una línea punteada vertical para cada variable
        ax.axhline(y=var_name, color='gray', linestyle='--', linewidth=0.6)

    # Personalizar el eje y para que se ajuste a los nombres de las variables
    ax.set_yticklabels(bootstrap_df['var_name'])

    # Personalizar el eje X con más valores
    ax.set_xlim(-2, 2)  # Establecer límites personalizados para el eje X
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])  # Definir los valores de las marcas en el eje X

    # Agregar una línea punteada en el eje x en el valor 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Crear la leyenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Not melanoma lesions', markerfacecolor='blue', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Melanoma lesions', markerfacecolor='green', markersize=6)]

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.07), fontsize=9)

    # fig.subplots_adjust(right=0.1)

    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    plt.show()


def generate_feature_importance_plot(df, file_path):
    df_sorted = df.sort_values(by='score', ascending=False)
    # Selecciona las características principales
    # top_features = df_sorted.head(top_n_features)
    top_features = df_sorted
    # Crea el gráfico de barras
    plt.figure(figsize=(220, 100))
    plt.bar(top_features['var_name'].astype(str), top_features['score'])
    plt.ylabel('Feature importance')
    plt.xlabel('Features')

    # Rotación de etiquetas del eje x para mejor visualización
    plt.xticks(rotation=90)
    # plt.tight_layout()

    # Guarda el gráfico como una imagen
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def generate_graphic_metrics(df, filepath, type_sampling):

    df["Model Name"] = df["model"].str.split("+").str[1]

    over_df = df[df["type_sampling"] == type_sampling]
    unique_models = over_df["Model Name"].unique()
    unique_metrics = over_df["eval_metric"].unique()
    colors = ['b', 'orange', 'g', 'r', 'm', 'y', 'k']
    markers = ['o', "^", 'x', "s"]

    for metric in unique_metrics:
        print(metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_size_inches(6, 3)

        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        #ax.set_title(f"{metric} by IR and Model")
        ax.set_xlabel("IMBALANCE RATIO")
        ax.set_ylabel(f"{metric.upper()}")    
        for i, model in enumerate(unique_models):
            subset = over_df[(over_df["Model Name"] == model) & (over_df["eval_metric"] == metric)]
            ir_values = subset["model"].str.split("+").str[-2].astype(float)
            means = subset["mean"]
            stds = subset["std"]
            ax.plot(ir_values, means, label=model.upper(), marker=markers[i], markerfacecolor ='none', linestyle='-', color=colors[i])
            ax.fill_between(ir_values, means - stds, means + stds, color=colors[i], alpha=0.15)
        
        #ax.set_ylim([0.6, 1])  # Personalizar el rango del eje y
        ax.grid(alpha=0.5, linestyle='--')  # Personalizar la cuadrícula
        ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))  # Agregar una leyenda
        plt.tight_layout()

        filepath_new = filepath.replace(f'graphic_', f'{metric}_graphic_')
        print(filepath_new)

        plt.savefig(filepath_new, format='pdf')
        
        plt.close()


