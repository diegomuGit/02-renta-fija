from enum import Enum
from typing import List, Tuple, Optional
from datetime import date
import numpy as np
import pandas as pd


def interpolar_factor_descuento(
    fecha_flujo: date,
    df_curva: pd.DataFrame,
    fecha_valoracion: date
) -> float:
    """
    Interpola el factor de descuento para una fecha dada utilizando interpolación log-lineal.
    
    Si la fecha coincide exactamente con un punto de la curva, retorna el factor de descuento directo.
    Si no, interpola log-linealmente entre los dos puntos más cercanos.
    
    Args:
        fecha_flujo: Fecha del flujo a descontar
        df_curva: DataFrame con la curva ESTR (Date, Discount)
        fecha_valoracion: Fecha de valoración (origen)
    
    Returns:
        float: Factor de descuento para la fecha del flujo
    """
    
    # Convertir fecha_flujo a datetime para comparar
    fecha_flujo_dt = pd.Timestamp(fecha_flujo)
    
    # Buscar coincidencia exacta
    coincidencia = df_curva[df_curva['Date'] == fecha_flujo_dt]
    if not coincidencia.empty:
        return float(coincidencia['Discount'].iloc[0])
    
    # No hay coincidencia exacta: interpolar
    # Encontrar el punto inmediatamente anterior e inmediatamente posterior
    puntos_anteriores = df_curva[df_curva['Date'] < fecha_flujo_dt]
    puntos_posteriores = df_curva[df_curva['Date'] > fecha_flujo_dt]
    
    # Validaciones
    if puntos_anteriores.empty:
        # El flujo es anterior al primer punto de la curva
        # Usar el primer factor de descuento disponible
        return float(df_curva['Discount'].iloc[0])
    
    if puntos_posteriores.empty:
        # El flujo es posterior al último punto de la curva
        # Extrapolar usando el último punto
        return float(df_curva['Discount'].iloc[-1])
    
    # Obtener los dos puntos para interpolar
    punto_1 = puntos_anteriores.iloc[-1]  # Punto más cercano anterior
    punto_2 = puntos_posteriores.iloc[0]  # Punto más cercano posterior
    
    # Extraer fechas y factores de descuento
    t1 = punto_1['Date']
    t2 = punto_2['Date']
    df1 = float(punto_1['Discount'])
    df2 = float(punto_2['Discount'])
    
    # Calcular plazos en años (base ACT/365)
    T1 = (t1 - pd.Timestamp(fecha_valoracion)).days / 365.0
    T2 = (t2 - pd.Timestamp(fecha_valoracion)).days / 365.0
    T_flujo = (fecha_flujo_dt - pd.Timestamp(fecha_valoracion)).days / 365.0
    
    # Interpolación log-lineal de factores de descuento:
    # ln(DF(T)) = ln(DF(T1)) + (ln(DF(T2)) - ln(DF(T1))) * (T - T1) / (T2 - T1)
    
    log_df1 = np.log(df1)
    log_df2 = np.log(df2)
    
    log_df_flujo = log_df1 + (log_df2 - log_df1) * (T_flujo - T1) / (T2 - T1)
    
    df_flujo = np.exp(log_df_flujo)
    
    return df_flujo


class MetodoDescuento(Enum):
    """Métodos de descuento disponibles"""
    CURVA = "curva"  # Usando curva de tipos + spread
    YIELD = "yield"   # Usando yield único constante


def calcular_valor_presente(
    flujos_caja,  # List[Tuple[date, float]] o List[Tuple[float, float]]
    fecha_valoracion: Optional[date] = None,
    metodo: MetodoDescuento = MetodoDescuento.YIELD,
    # Parámetros para método CURVA
    df_curva: Optional[pd.DataFrame] = None,
    spread_credito: float = 0.0,
    # Parámetros para método YIELD
    yield_anual: Optional[float] = None,
    frecuencia: int = 2,
    # Formato de flujos
    flujos_como_fechas: bool = True
) -> float:
    """
    Calcula el valor presente de flujos de caja usando diferentes métodos de descuento.
    
    Args:
        flujos_caja: Lista de tuplas (fecha, monto) si flujos_como_fechas=True
                     o (tiempo_años, monto) si flujos_como_fechas=False
        fecha_valoracion: Fecha de valoración (requerido si flujos_como_fechas=True)
        metodo: Método de descuento (CURVA o YIELD)
        
        # Para método CURVA (valoración mark-to-market):
        df_curva: DataFrame con la curva ESTR (columnas: 'Date', 'Discount')
        spread_credito: Spread de crédito en decimal (ej: 0.01 para 100 bps)
        
        # Para método YIELD (cálculo de YTM):
        yield_anual: Yield anual en decimal (ej: 0.03 para 3%)
        frecuencia: Frecuencia de capitalización (1=anual, 2=semestral, 4=trimestral)
        
        # Formato de flujos:
        flujos_como_fechas: Si True, flujos_caja contiene (fecha, monto)
                           Si False, flujos_caja contiene (tiempo_años, monto)
    
    Returns:
        float: Valor presente de los flujos
        
    Raises:
        ValueError: Si faltan parámetros requeridos según el método elegido
        
    Examples:
        >>> # Valoración con curva de mercado
        >>> vp = calcular_valor_presente(
        ...     flujos, fecha_val, MetodoDescuento.CURVA,
        ...     df_curva=curva_estr, spread_credito=0.01
        ... )
        
        >>> # Valoración con yield único (flujos como fechas)
        >>> vp = calcular_valor_presente(
        ...     flujos, fecha_val, MetodoDescuento.YIELD,
        ...     yield_anual=0.03, frecuencia=2
        ... )
        
        >>> # Valoración con yield único (flujos como tiempos)
        >>> vp = calcular_valor_presente(
        ...     flujos_tiempo, metodo=MetodoDescuento.YIELD,
        ...     yield_anual=0.03, frecuencia=2, flujos_como_fechas=False
        ... )
    """
    
    if metodo == MetodoDescuento.CURVA:
        # ============ MÉTODO CURVA ============
        if df_curva is None:
            raise ValueError("Se requiere 'df_curva' para el método CURVA")
        if not flujos_como_fechas:
            raise ValueError("El método CURVA requiere flujos_como_fechas=True")
        if fecha_valoracion is None:
            raise ValueError("Se requiere 'fecha_valoracion' para el método CURVA")
        
        valor_presente = 0.0
        
        for fecha_flujo, monto_flujo in flujos_caja:
            # FILTRAR: Solo descontar flujos FUTUROS (>= fecha de valoración)
            if fecha_flujo < fecha_valoracion:
                continue  # Saltar flujos pasados
            
            # Obtener factor de descuento de la curva (interpolado si es necesario)
            df_curva_flujo = interpolar_factor_descuento(
                fecha_flujo, df_curva, fecha_valoracion
            )
            
            # Calcular plazo en años (ACT/365)
            plazo_anos = (fecha_flujo - fecha_valoracion).days / 365.0
            
            # Aplicar spread de crédito con capitalización continua
            # DF_ajustado = DF_curva × exp(-spread × T)
            factor_descuento_ajustado = df_curva_flujo * np.exp(-spread_credito * plazo_anos)
            
            # Descontar el flujo
            valor_presente += monto_flujo * factor_descuento_ajustado
        
        return valor_presente
    
    elif metodo == MetodoDescuento.YIELD:
        # ============ MÉTODO YIELD ============
        if yield_anual is None:
            raise ValueError("Se requiere 'yield_anual' para el método YIELD")
        
        valor_presente = 0.0
        
        if flujos_como_fechas:
            # Flujos en formato (fecha, monto)
            if fecha_valoracion is None:
                raise ValueError("Se requiere 'fecha_valoracion' cuando flujos_como_fechas=True")
            
            for fecha_flujo, monto_flujo in flujos_caja:
                # FILTRAR: Solo descontar flujos FUTUROS (>= fecha de valoración)
                if fecha_flujo < fecha_valoracion:
                    continue  # Saltar flujos pasados
                
                # Calcular tiempo en años (ACT/365)
                tiempo_anos = (fecha_flujo - fecha_valoracion).days / 365.0
                
                # Descuento con composición periódica
                # DF = 1 / (1 + y/freq)^(t × freq)
                factor_descuento = (1 + yield_anual / frecuencia) ** (tiempo_anos * frecuencia)
                valor_presente += monto_flujo / factor_descuento
        else:
            # Flujos en formato (tiempo_años, monto)
            for tiempo_anos, monto_flujo in flujos_caja:
                # FILTRAR: Solo descontar flujos FUTUROS (tiempo > 0)
                if tiempo_anos <= 0:
                    continue  # Saltar flujos pasados o en el presente
                
                # Descuento con composición periódica
                # DF = 1 / (1 + y/freq)^(t × freq)
                factor_descuento = (1 + yield_anual / frecuencia) ** (tiempo_anos * frecuencia)
                valor_presente += monto_flujo / factor_descuento
        
        return valor_presente
    
    else:
        raise ValueError(f"Método de descuento no reconocido: {metodo}")
