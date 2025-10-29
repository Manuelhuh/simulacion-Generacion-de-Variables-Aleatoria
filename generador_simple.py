import math
import random
import collections
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class GeneradorVariablesSimple:
    def __init__(self, semilla=42):
        random.seed(semilla)
    
    def transformada_inversa_exponencial(self, lambd: float, tamaño: int = 1000) -> List[float]:
        """
        Método de la transformada inversa para distribución exponencial
        F⁻¹(u) = -ln(1-u)/λ
        """
        datos = []
        for _ in range(tamaño):
            u = random.random()
            x = -math.log(1 - u) / lambd
            datos.append(x)
        return datos
    
    def convolucion_binomial(self, n: int, p: float, tamaño: int = 1000) -> List[int]:
        """
        Método de convolución para distribución binomial
        Suma de n variables Bernoulli independientes
        """
        datos = []
        for _ in range(tamaño):
            # Sumar n ensayos Bernoulli
            exitos = sum(1 for _ in range(n) if random.random() < p)
            datos.append(exitos)
        return datos
    
    def generar_poisson(self, lambd: float, tamaño: int = 1000) -> List[int]:
        """
        Generación de distribución Poisson usando el método de Knuth
        """
        datos = []
        for _ in range(tamaño):
            L = math.exp(-lambd)
            k = 0
            p = 1.0
            
            while p > L:
                k += 1
                p *= random.random()
            
            datos.append(k - 1)
        return datos
    
    def generar_normal_box_muller(self, mu: float, sigma: float, tamaño: int = 1000) -> List[float]:
        """
        Generación de distribución normal usando Box-Muller
        """
        datos = []
        for i in range(0, tamaño, 2):
            u1 = random.random()
            u2 = random.random()
            
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
            
            datos.append(mu + sigma * z0)
            if i + 1 < tamaño:
                datos.append(mu + sigma * z1)
        
        return datos[:tamaño]

class AnalizadorSimple:
    def __init__(self):
        self.generador = GeneradorVariablesSimple()
    
    def calcular_estadisticos(self, datos: List[float]) -> Dict:
        """Calcula estadísticos descriptivos básicos"""
        n = len(datos)
        media = sum(datos) / n
        varianza = sum((x - media) ** 2 for x in datos) / (n - 1)
        desviacion = math.sqrt(varianza)
        
        return {
            'n': n,
            'media': media,
            'varianza': varianza,
            'desviacion': desviacion,
            'minimo': min(datos),
            'maximo': max(datos)
        }
    
    def prueba_frecuencias(self, observados: List[int], esperados: List[float]) -> float:
        """
        Prueba Chi-cuadrado simplificada para bondad de ajuste
        """
        chi2 = 0.0
        for obs, esp in zip(observados, esperados):
            if esp > 0:  # Evitar división por cero
                chi2 += (obs - esp) ** 2 / esp
        return chi2
    
    def visualizar_histograma(self, datos: List[float], titulo: str, bins: int = 30):
        """
        Visualiza histograma simple de los datos
        """
        plt.figure(figsize=(10, 6))
        
        # Para datos continuos
        if isinstance(datos[0], float):
            plt.hist(datos, bins=bins, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            plt.ylabel('Densidad')
        else:
            # Para datos discretos
            conteo = collections.Counter(datos)
            valores = list(conteo.keys())
            frecuencias = list(conteo.values())
            plt.bar(valores, frecuencias, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.ylabel('Frecuencia')
        
        plt.xlabel('Valor')
        plt.title(titulo)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def comparar_metodos(self, datos1: List[float], datos2: List[float], 
                        titulo1: str, titulo2: str):
        """
        Compara dos conjuntos de datos generados por diferentes métodos
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histograma del primer método
        ax1.hist(datos1, bins=30, density=True, alpha=0.7, color='blue')
        ax1.set_title(titulo1)
        ax1.set_xlabel('Valor')
        ax1.set_ylabel('Densidad')
        ax1.grid(True, alpha=0.3)
        
        # Histograma del segundo método
        ax2.hist(datos2, bins=30, density=True, alpha=0.7, color='red')
        ax2.set_title(titulo2)
        ax2.set_xlabel('Valor')
        ax2.set_ylabel('Densidad')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar estadísticos comparativos
        stats1 = self.calcular_estadisticos(datos1)
        stats2 = self.calcular_estadisticos(datos2)
        
        print(f"\nComparación de estadísticos:")
        print(f"{'Estadístico':<15} {titulo1:<20} {titulo2:<20}")
        print("-" * 60)
        for key in ['media', 'varianza', 'desviacion']:
            print(f"{key:<15} {stats1[key]:<20.4f} {stats2[key]:<20.4f}")

def distribucion_binomial_teorica(k: int, n: int, p: float) -> float:
    """Calcula P(X=k) para distribución binomial teórica"""
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1 - p) ** (n - k))

def distribucion_poisson_teorica(k: int, lambd: float) -> float:
    """Calcula P(X=k) para distribución Poisson teórica"""
    return (math.exp(-lambd) * (lambd ** k)) / math.factorial(k)

def main():
    analizador = AnalizadorSimple()
    
    print("=== GENERADOR DE VARIABLES ALEATORIAS (Versión Simple) ===")
    print("Distribuciones disponibles:")
    print("1. Binomial (Convolución)")
    print("2. Poisson")
    print("3. Exponencial (Transformada Inversa)")
    print("4. Normal (Box-Muller)")
    
    while True:
        try:
            opcion = input("\nSeleccione distribución (1-4) o 'q' para salir: ").strip()
            
            if opcion.lower() == 'q':
                break
            
            opcion = int(opcion)
            tamaño = 1000  # Tamaño de muestra por defecto
            
            if opcion == 1:
                print("\n--- DISTRIBUCIÓN BINOMIAL ---")
                n = int(input("Ingrese n (número de ensayos): "))
                p = float(input("Ingrese p (probabilidad de éxito): "))
                
                datos = analizador.generador.convolucion_binomial(n, p, tamaño)
                analizador.visualizar_histograma(datos, f'Binomial(n={n}, p={p})')
                
                # Estadísticos
                stats = analizador.calcular_estadisticos(datos)
                print(f"\nEstadísticos de la muestra:")
                for key, value in stats.items():
                    print(f"{key}: {value:.4f}")
                
                # Comparación teórica
                print(f"\nComparación con teórica (μ=np={n*p:.2f}, σ²=np(1-p)={n*p*(1-p):.2f}):")
                print(f"Media muestral: {stats['media']:.4f}")
                print(f"Varianza muestral: {stats['varianza']:.4f}")
                
            elif opcion == 2:
                print("\n--- DISTRIBUCIÓN POISSON ---")
                lambd = float(input("Ingrese λ (tasa promedio): "))
                
                datos = analizador.generador.generar_poisson(lambd, tamaño)
                analizador.visualizar_histograma(datos, f'Poisson(λ={lambd})')
                
                stats = analizador.calcular_estadisticos(datos)
                print(f"\nEstadísticos de la muestra:")
                for key, value in stats.items():
                    print(f"{key}: {value:.4f}")
                
                print(f"\nComparación con teórica (μ=λ={lambd:.2f}, σ²=λ={lambd:.2f}):")
                print(f"Media muestral: {stats['media']:.4f}")
                print(f"Varianza muestral: {stats['varianza']:.4f}")
                
            elif opcion == 3:
                print("\n--- DISTRIBUCIÓN EXPONENCIAL ---")
                lambd = float(input("Ingrese λ (tasa): "))
                
                # Generar con transformada inversa
                datos_ti = analizador.generador.transformada_inversa_exponencial(lambd, tamaño)
                
                # Generar con método alternativo (transformada inversa con u en [0,1))
                datos_alt = [-math.log(random.random()) / lambd for _ in range(tamaño)]
                
                analizador.comparar_metodos(
                    datos_ti, datos_alt,
                    f'Exponencial TI (λ={lambd})',
                    f'Exponencial Alternativo (λ={lambd})'
                )
                
                stats_ti = analizador.calcular_estadisticos(datos_ti)
                stats_alt = analizador.calcular_estadisticos(datos_alt)
                
                print(f"\nComparación con teórica (μ=1/λ={1/lambd:.2f}, σ²=1/λ²={1/(lambd**2):.2f}):")
                print("Método Transformada Inversa:")
                print(f"  Media: {stats_ti['media']:.4f}, Varianza: {stats_ti['varianza']:.4f}")
                print("Método Alternativo:")
                print(f"  Media: {stats_alt['media']:.4f}, Varianza: {stats_alt['varianza']:.4f}")
                
            elif opcion == 4:
                print("\n--- DISTRIBUCIÓN NORMAL ---")
                mu = float(input("Ingrese μ (media): "))
                sigma = float(input("Ingrese σ (desviación estándar): "))
                
                datos = analizador.generador.generar_normal_box_muller(mu, sigma, tamaño)
                analizador.visualizar_histograma(datos, f'Normal(μ={mu}, σ={sigma})')
                
                stats = analizador.calcular_estadisticos(datos)
                print(f"\nEstadísticos de la muestra:")
                for key, value in stats.items():
                    print(f"{key}: {value:.4f}")
                
                print(f"\nComparación con teórica (μ={mu:.2f}, σ²={sigma**2:.2f}):")
                print(f"Media muestral: {stats['media']:.4f}")
                print(f"Varianza muestral: {stats['varianza']:.4f}")
                
            else:
                print("Opción no válida. Intente nuevamente.")
                
        except ValueError as e:
            print(f"Error en entrada de datos: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()