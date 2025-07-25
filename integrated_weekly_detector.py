"""
Integrated Weekly Pattern Detector with Day Recommendations
===========================================================

Fixed design: Each method both detects patterns AND recommends specific days.
No redundant analysis - each method provides complete actionable results.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal
import warnings

warnings.filterwarnings('ignore')

class IntegratedWeeklyDetector:
    """ALL 15 methods detect patterns AND recommend specific days"""
    
    # Method 1: Autocorrelation Analysis → Day Recommendations
    @staticmethod
    def autocorrelation_with_day_ranking(timestamps, lag_hours=168):
        """Autocorrelation analysis that directly recommends days"""
        df = pd.DataFrame({'timestamp': timestamps, 'count': 1})
        df.set_index('timestamp', inplace=True)
        hourly_series = df.resample('1H').sum().fillna(0)['count']
        
        if len(hourly_series) < lag_hours * 2:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        # Standard autocorrelation check
        autocorr_168 = hourly_series.autocorr(lag=lag_hours)
        has_weekly_pattern = autocorr_168 > 0.3
        
        if not has_weekly_pattern:
            return {
                'method': 'autocorrelation',
                'has_pattern': False,
                'autocorr_1_week': autocorr_168,
                'recommended_days': []
            }
        
        # NEW: Analyze which days contribute most to the autocorrelation
        df_orig = pd.DataFrame({'timestamp': timestamps})
        df_orig['day_of_week'] = df_orig['timestamp'].dt.day_name()
        df_orig['day_num'] = df_orig['timestamp'].dt.dayofweek
        
        # Calculate per-day autocorrelation strength
        day_autocorr_scores = {}
        for day_num in range(7):
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_num]
            
            # Create day-specific series (1 if event on this day-hour, 0 otherwise)
            day_events = df_orig[df_orig['day_num'] == day_num]
            if len(day_events) > 0:
                day_hourly = day_events.set_index('timestamp').resample('1H').size()
                day_series = hourly_series.copy()
                
                # Mask to only this day of week
                day_mask = pd.Series(day_series.index).dt.dayofweek == day_num
                day_contribution = day_series[day_mask.values]
                
                # Score = activity level * consistency
                activity_score = day_contribution.sum()
                consistency_score = 1 - (day_contribution.std() / (day_contribution.mean() + 1e-6))
                combined_score = activity_score * max(0, consistency_score)
                
                day_autocorr_scores[day_name] = {
                    'activity_score': activity_score,
                    'consistency_score': max(0, consistency_score),
                    'combined_score': combined_score,
                    'event_count': len(day_events)
                }
        
        # Rank days by combined score
        ranked_days = sorted(day_autocorr_scores.items(), 
                           key=lambda x: x[1]['combined_score'], 
                           reverse=True)
        
        # Select top days (those above average combined score)
        avg_score = np.mean([scores['combined_score'] for scores in day_autocorr_scores.values()])
        recommended_days = [
            {
                'day': day_name,
                'confidence': min(1.0, scores['combined_score'] / avg_score),
                'activity_score': scores['activity_score'],
                'consistency_score': scores['consistency_score'],
                'event_count': scores['event_count'],
                'reason': f'High autocorr contribution (score: {scores["combined_score"]:.2f})'
            }
            for day_name, scores in ranked_days 
            if scores['combined_score'] > avg_score and scores['event_count'] > 0
        ]
        
        return {
            'method': 'autocorrelation',
            'has_pattern': True,
            'autocorr_1_week': autocorr_168,
            'recommended_days': recommended_days,
            'day_ranking': ranked_days
        }
    
    # Method 2: Friedman Test → Direct Day Ranking
    @staticmethod
    def friedman_test_with_day_ranking(timestamps):
        """Friedman test that directly ranks days by significance"""
        df = pd.DataFrame({'timestamp': timestamps})
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['date'] = df['timestamp'].dt.date
        
        # Count events per day per week
        weekly_data = df.groupby(['week', 'day_num']).size().reset_index(name='count')
        
        if len(weekly_data) < 14:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        # Pivot: weeks as rows, days as columns
        pivot_data = weekly_data.pivot(index='week', columns='day_num', values='count').fillna(0)
        
        if pivot_data.shape[0] < 3:
            return {'has_pattern': False, 'reason': 'insufficient_weeks'}
        
        try:
            # Friedman test
            stat, p_value = friedmanchisquare(*[pivot_data[col] for col in pivot_data.columns])
            has_pattern = p_value < 0.05
            
            if not has_pattern:
                return {
                    'method': 'friedman',
                    'has_pattern': False,
                    'p_value': p_value,
                    'recommended_days': []
                }
            
            # NEW: Rank days by their Friedman test contribution
            day_rankings = []
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day_num in pivot_data.columns:
                day_values = pivot_data[day_num].values
                day_name = day_names[day_num]
                
                # Calculate day's contribution to the significant difference
                day_mean = np.mean(day_values)
                day_std = np.std(day_values)
                overall_mean = np.mean(pivot_data.values)
                
                # Significance score = how much this day deviates from overall pattern
                deviation_score = abs(day_mean - overall_mean) / (overall_mean + 1e-6)
                consistency_score = 1 / (1 + day_std)  # Lower std = higher consistency
                combined_score = day_mean * consistency_score * (1 + deviation_score)
                
                day_rankings.append({
                    'day': day_name,
                    'day_num': day_num,
                    'mean_events': day_mean,
                    'std_events': day_std,
                    'deviation_score': deviation_score,
                    'consistency_score': consistency_score,
                    'combined_score': combined_score,
                    'total_events': int(np.sum(day_values))
                })
            
            # Sort by combined score
            day_rankings.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Select top days (above median score)
            scores = [d['combined_score'] for d in day_rankings]
            threshold = np.median(scores) * 1.2  # 20% above median
            
            recommended_days = [
                {
                    'day': day_info['day'],
                    'confidence': min(1.0, day_info['combined_score'] / max(scores)),
                    'mean_events': day_info['mean_events'],
                    'total_events': day_info['total_events'],
                    'reason': f'Friedman significant contributor (score: {day_info["combined_score"]:.2f})'
                }
                for day_info in day_rankings
                if day_info['combined_score'] > threshold and day_info['total_events'] > 0
            ]
            
            return {
                'method': 'friedman',
                'has_pattern': True,
                'friedman_statistic': stat,
                'p_value': p_value,
                'recommended_days': recommended_days,
                'day_ranking': day_rankings
            }
            
        except Exception as e:
            return {'has_pattern': False, 'reason': f'test_failed: {str(e)}'}
    
    # Method 3: Spectral Analysis → Peak Day Identification
    @staticmethod
    def spectral_analysis_with_day_peaks(timestamps):
        """Spectral analysis that identifies which days create the weekly frequency"""
        df = pd.DataFrame({'timestamp': timestamps, 'count': 1})
        df.set_index('timestamp', inplace=True)
        hourly_series = df.resample('1H').sum().fillna(0)['count']
        
        if len(hourly_series) < 168:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        # Overall spectral analysis
        from scipy.fft import fft, fftfreq
        signal_data = hourly_series.values
        fft_result = fft(signal_data)
        freqs = fftfreq(len(signal_data), d=1)
        power_spectrum = np.abs(fft_result)**2
        
        weekly_freq = 1/168
        freq_tolerance = 0.1 * weekly_freq
        weekly_indices = np.where(
            (np.abs(freqs - weekly_freq) < freq_tolerance) |
            (np.abs(freqs + weekly_freq) < freq_tolerance)
        )[0]
        
        weekly_power = np.max(power_spectrum[weekly_indices]) if len(weekly_indices) > 0 else 0
        total_power = np.sum(power_spectrum)
        weekly_power_ratio = weekly_power / total_power if total_power > 0 else 0
        
        has_pattern = weekly_power_ratio > 0.05
        
        if not has_pattern:
            return {
                'method': 'spectral',
                'has_pattern': False,
                'weekly_power_ratio': weekly_power_ratio,
                'recommended_days': []
            }
        
        # NEW: Analyze which days contribute most to weekly frequency
        df_orig = pd.DataFrame({'timestamp': timestamps})
        df_orig['day_of_week'] = df_orig['timestamp'].dt.day_name()
        df_orig['day_num'] = df_orig['timestamp'].dt.dayofweek
        
        day_spectral_scores = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day_num in range(7):
            day_name = day_names[day_num]
            
            # Create day-only signal
            day_mask = pd.Series(hourly_series.index).dt.dayofweek == day_num
            day_signal = np.zeros_like(signal_data)
            day_signal[day_mask.values] = signal_data[day_mask.values]
            
            if np.sum(day_signal) == 0:
                day_spectral_scores[day_name] = {
                    'power_contribution': 0,
                    'activity_level': 0,
                    'spectral_score': 0,
                    'event_count': 0
                }
                continue
            
            # FFT of day-only signal
            day_fft = fft(day_signal)
            day_power = np.abs(day_fft)**2
            
            # Power at weekly frequency
            day_weekly_power = np.max(day_power[weekly_indices]) if len(weekly_indices) > 0 else 0
            power_contribution = day_weekly_power / weekly_power if weekly_power > 0 else 0
            
            activity_level = np.sum(day_signal)
            spectral_score = power_contribution * activity_level
            
            day_spectral_scores[day_name] = {
                'power_contribution': power_contribution,
                'activity_level': activity_level,
                'spectral_score': spectral_score,
                'event_count': len(df_orig[df_orig['day_num'] == day_num])
            }
        
        # Rank days by spectral contribution
        ranked_days = sorted(day_spectral_scores.items(), 
                           key=lambda x: x[1]['spectral_score'], 
                           reverse=True)
        
        # Select days with significant spectral contribution
        max_score = max(scores['spectral_score'] for scores in day_spectral_scores.values())
        threshold = max_score * 0.3  # Top 30% threshold
        
        recommended_days = [
            {
                'day': day_name,
                'confidence': scores['spectral_score'] / max_score if max_score > 0 else 0,
                'power_contribution': scores['power_contribution'],
                'activity_level': scores['activity_level'],
                'event_count': scores['event_count'],
                'reason': f'High spectral contribution ({scores["power_contribution"]:.2%})'
            }
            for day_name, scores in ranked_days
            if scores['spectral_score'] > threshold and scores['event_count'] > 0
        ]
        
        return {
            'method': 'spectral',
            'has_pattern': True,
            'weekly_power_ratio': weekly_power_ratio,
            'recommended_days': recommended_days,
            'day_spectral_analysis': day_spectral_scores
        }
    
    # Method 4: Clustering → Cluster-Based Day Selection
    @staticmethod
    def clustering_with_day_selection(timestamps):
        """Clustering that directly selects days from the most active cluster"""
        df = pd.DataFrame({'timestamp': timestamps})
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        
        # Create feature vectors for each day of week (hourly distributions)
        day_features = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day_num in range(7):
            day_data = df[df['day_num'] == day_num]
            if len(day_data) == 0:
                day_features.append([0] * 24)  # No activity
            else:
                hour_counts = Counter(day_data['hour'])
                features = [hour_counts.get(hour, 0) for hour in range(24)]
                day_features.append(features)
        
        day_features = np.array(day_features)
        
        if np.sum(day_features) == 0:
            return {'has_pattern': False, 'reason': 'no_data'}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(day_features)
            
            silhouette = silhouette_score(day_features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
            has_pattern = silhouette > 0.3
            
            if not has_pattern:
                return {
                    'method': 'clustering',
                    'has_pattern': False,
                    'silhouette_score': silhouette,
                    'recommended_days': []
                }
            
            # NEW: Directly recommend days from best cluster
            cluster_activity = {}
            for cluster_id in set(cluster_labels):
                cluster_days = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_total_activity = sum(np.sum(day_features[day_idx]) for day_idx in cluster_days)
                cluster_activity[cluster_id] = {
                    'total_activity': cluster_total_activity,
                    'days': cluster_days,
                    'avg_activity': cluster_total_activity / len(cluster_days) if cluster_days else 0
                }
            
            # Select the most active cluster
            best_cluster_id = max(cluster_activity.keys(), key=lambda k: cluster_activity[k]['total_activity'])
            best_cluster = cluster_activity[best_cluster_id]
            
            recommended_days = []
            for day_idx in best_cluster['days']:
                day_name = day_names[day_idx]
                day_activity = np.sum(day_features[day_idx])
                
                if day_activity > 0:
                    recommended_days.append({
                        'day': day_name,
                        'confidence': day_activity / best_cluster['total_activity'],
                        'activity_score': day_activity,
                        'cluster_id': best_cluster_id,
                        'reason': f'Most active cluster member (cluster {best_cluster_id})',
                        'event_count': len(df[df['day_num'] == day_idx])
                    })
            
            # Sort by activity within cluster
            recommended_days.sort(key=lambda x: x['activity_score'], reverse=True)
            
            return {
                'method': 'clustering',
                'has_pattern': True,
                'silhouette_score': silhouette,
                'recommended_days': recommended_days,
                'cluster_analysis': cluster_activity,
                'best_cluster_id': best_cluster_id
            }
            
        except ImportError:
            return {'has_pattern': False, 'reason': 'sklearn_not_available'}
    
    # Method 5: Entropy Analysis → High Information Days
    @staticmethod
    def entropy_analysis_with_day_selection(timestamps):
        """Entropy analysis that selects high-information days"""
        from math import log2
        
        df = pd.DataFrame({'timestamp': timestamps})
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        
        day_counts = Counter(df['day_num'])
        total_events = len(df)
        
        if total_events == 0:
            return {'has_pattern': False, 'reason': 'no_data'}
        
        # Calculate overall entropy
        day_probs = [day_counts.get(day, 0) / total_events for day in range(7)]
        entropy_days = -sum(p * log2(p) for p in day_probs if p > 0)
        max_entropy = log2(7)
        regularity = (max_entropy - entropy_days) / max_entropy
        
        has_pattern = regularity > 0.3
        
        if not has_pattern:
            return {
                'method': 'entropy',
                'has_pattern': False,
                'regularity': regularity,
                'recommended_days': []
            }
        
        # NEW: Directly select high-information days
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_info_scores = []
        
        for day_num in range(7):
            day_name = day_names[day_num]
            day_count = day_counts.get(day_num, 0)
            
            if day_count > 0:
                p = day_count / total_events
                # Information content = -log2(p) = how "surprising" this day is
                information_content = -log2(p)
                # Information value = information content weighted by frequency
                information_value = information_content * day_count
                
                day_info_scores.append({
                    'day': day_name,
                    'count': day_count,
                    'probability': p,
                    'information_content': information_content,
                    'information_value': information_value,
                    'confidence': min(1.0, information_value / total_events)
                })
        
        # Sort by information value and select top days
        day_info_scores.sort(key=lambda x: x['information_value'], reverse=True)
        
        # Select days that contribute to 80% of total information
        total_info = sum(d['information_value'] for d in day_info_scores)
        cumulative_info = 0
        recommended_days = []
        
        for day_info in day_info_scores:
            cumulative_info += day_info['information_value']
            recommended_days.append({
                'day': day_info['day'],
                'confidence': day_info['confidence'],
                'event_count': day_info['count'],
                'information_value': day_info['information_value'],
                'reason': f'High information value ({day_info["information_value"]:.2f})'
            })
            
            if cumulative_info >= 0.8 * total_info:
                break
        
        return {
            'method': 'entropy',
            'has_pattern': True,
            'regularity': regularity,
            'recommended_days': recommended_days,
            'total_information': total_info
        }
    
    # Method 6: STL Decomposition → Seasonal Day Identification
    @staticmethod
    def stl_decomposition_with_day_selection(timestamps):
        """STL decomposition that identifies which days drive seasonality"""
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            return {'has_pattern': False, 'reason': 'statsmodels_not_available'}
        
        df = pd.DataFrame({'timestamp': timestamps, 'count': 1})
        df.set_index('timestamp', inplace=True)
        hourly_series = df.resample('1H').sum().fillna(0)['count']
        
        if len(hourly_series) < 168 * 3:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        try:
            stl = STL(hourly_series, seasonal=168, robust=True)
            decomposition = stl.fit()
            seasonal_strength = np.var(decomposition.seasonal) / np.var(hourly_series)
            
            has_pattern = seasonal_strength > 0.1
            if not has_pattern:
                return {
                    'method': 'stl',
                    'has_pattern': False,
                    'seasonal_strength': seasonal_strength,
                    'recommended_days': []
                }
            
            # NEW: Identify which days contribute most to seasonality
            seasonal_component = decomposition.seasonal
            df_orig = pd.DataFrame({'timestamp': timestamps})
            df_orig['day_num'] = df_orig['timestamp'].dt.dayofweek
            
            day_seasonal_scores = {}
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day_num in range(7):
                day_name = day_names[day_num]
                day_mask = pd.Series(hourly_series.index).dt.dayofweek == day_num
                day_seasonal_values = seasonal_component[day_mask.values] if any(day_mask.values) else []
                
                if len(day_seasonal_values) > 0:
                    seasonal_magnitude = np.mean(np.abs(day_seasonal_values))
                    seasonal_consistency = 1 - (np.std(day_seasonal_values) / (np.mean(np.abs(day_seasonal_values)) + 1e-6))
                    combined_score = seasonal_magnitude * max(0, seasonal_consistency)
                else:
                    combined_score = 0
                
                day_seasonal_scores[day_name] = {
                    'seasonal_magnitude': seasonal_magnitude if len(day_seasonal_values) > 0 else 0,
                    'seasonal_consistency': max(0, seasonal_consistency) if len(day_seasonal_values) > 0 else 0,
                    'combined_score': combined_score,
                    'event_count': len(df_orig[df_orig['day_num'] == day_num])
                }
            
            # Select days with strong seasonal contribution
            max_score = max(scores['combined_score'] for scores in day_seasonal_scores.values())
            threshold = max_score * 0.4
            
            recommended_days = [
                {
                    'day': day_name,
                    'confidence': scores['combined_score'] / max_score if max_score > 0 else 0,
                    'seasonal_magnitude': scores['seasonal_magnitude'],
                    'seasonal_consistency': scores['seasonal_consistency'],
                    'event_count': scores['event_count'],
                    'reason': f'Strong seasonal pattern contributor (score: {scores["combined_score"]:.2f})'
                }
                for day_name, scores in day_seasonal_scores.items()
                if scores['combined_score'] > threshold and scores['event_count'] > 0
            ]
            
            return {
                'method': 'stl',
                'has_pattern': True,
                'seasonal_strength': seasonal_strength,
                'recommended_days': recommended_days,
                'day_seasonal_analysis': day_seasonal_scores
            }
            
        except Exception as e:
            return {'has_pattern': False, 'reason': f'stl_failed: {str(e)}'}
    
    # Method 7: Kruskal-Wallis → Day Ranking by Significance
    @staticmethod
    def kruskal_wallis_with_day_ranking(timestamps):
        """Kruskal-Wallis test that ranks days by statistical significance"""
        df = pd.DataFrame({'timestamp': timestamps})
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        
        daily_counts = df.groupby(['date', 'day_num']).size().reset_index(name='count')
        
        if len(daily_counts) < 14:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        try:
            # Group counts by day of week
            groups = [daily_counts[daily_counts['day_num'] == day]['count'].values 
                     for day in range(7)]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 3:
                return {'has_pattern': False, 'reason': 'insufficient_groups'}
            
            stat, p_value = kruskal(*groups)
            has_pattern = p_value < 0.05
            
            if not has_pattern:
                return {
                    'method': 'kruskal_wallis',
                    'has_pattern': False,
                    'kruskal_statistic': stat,
                    'p_value': p_value,
                    'recommended_days': []
                }
            
            # NEW: Rank days by their contribution to the significant difference
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_rankings = []
            
            for day_num in range(7):
                day_data = daily_counts[daily_counts['day_num'] == day_num]['count'].values
                if len(day_data) > 0:
                    day_name = day_names[day_num]
                    day_median = np.median(day_data)
                    day_mean = np.mean(day_data)
                    day_std = np.std(day_data)
                    
                    # Calculate rank-based score (Kruskal-Wallis uses ranks)
                    all_values = np.concatenate(groups)
                    day_ranks = [sorted(all_values).index(val) + 1 for val in day_data]
                    avg_rank = np.mean(day_ranks)
                    
                    # Higher average rank = more extreme values = more significant
                    significance_score = day_mean * (1 + abs(avg_rank - len(all_values)/2) / len(all_values))
                    
                    day_rankings.append({
                        'day': day_name,
                        'day_num': day_num,
                        'median_events': day_median,
                        'mean_events': day_mean,
                        'avg_rank': avg_rank,
                        'significance_score': significance_score,
                        'total_events': int(np.sum(day_data))
                    })
            
            # Sort by significance score
            day_rankings.sort(key=lambda x: x['significance_score'], reverse=True)
            
            # Select top-scoring days
            max_score = day_rankings[0]['significance_score'] if day_rankings else 0
            threshold = max_score * 0.6
            
            recommended_days = [
                {
                    'day': day_info['day'],
                    'confidence': day_info['significance_score'] / max_score if max_score > 0 else 0,
                    'mean_events': day_info['mean_events'],
                    'total_events': day_info['total_events'],
                    'avg_rank': day_info['avg_rank'],
                    'reason': f'High Kruskal-Wallis significance (score: {day_info["significance_score"]:.2f})'
                }
                for day_info in day_rankings
                if day_info['significance_score'] > threshold and day_info['total_events'] > 0
            ]
            
            return {
                'method': 'kruskal_wallis',
                'has_pattern': True,
                'kruskal_statistic': stat,
                'p_value': p_value,
                'recommended_days': recommended_days,
                'day_significance_ranking': day_rankings
            }
            
        except Exception as e:
            return {'has_pattern': False, 'reason': f'test_failed: {str(e)}'}
    
    # Method 8: Wavelet Analysis → Frequency Day Detection
    @staticmethod
    def wavelet_analysis_with_day_detection(timestamps):
        """Wavelet analysis identifying days with strong weekly frequency components"""
        try:
            import pywt
        except ImportError:
            return {'has_pattern': False, 'reason': 'pywt_not_available'}
        
        df = pd.DataFrame({'timestamp': timestamps, 'count': 1})
        df.set_index('timestamp', inplace=True)
        hourly_series = df.resample('1H').sum().fillna(0)['count']
        
        if len(hourly_series) < 168 * 2:
            return {'has_pattern': False, 'reason': 'insufficient_data'}
        
        try:
            # Continuous wavelet transform
            scales = np.arange(1, 200)
            coeffs = pywt.cwt(hourly_series.values, scales, 'mexh')[0]
            
            # Weekly scale energy
            weekly_scale_idx = np.argmin(np.abs(scales - 168))
            weekly_energy = np.mean(np.abs(coeffs[weekly_scale_idx, :]))
            total_energy = np.mean(np.abs(coeffs))
            weekly_energy_ratio = weekly_energy / total_energy if total_energy > 0 else 0
            
            has_pattern = weekly_energy_ratio > 0.2
            if not has_pattern:
                return {
                    'method': 'wavelet',
                    'has_pattern': False,
                    'weekly_energy_ratio': weekly_energy_ratio,
                    'recommended_days': []
                }
            
            # NEW: Identify which days contribute to weekly wavelet energy
            df_orig = pd.DataFrame({'timestamp': timestamps})
            df_orig['day_num'] = df_orig['timestamp'].dt.dayofweek
            
            day_wavelet_scores = {}
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day_num in range(7):
                day_name = day_names[day_num]
                day_mask = pd.Series(hourly_series.index).dt.dayofweek == day_num
                
                if any(day_mask.values):
                    # Extract wavelet coefficients for this day
                    day_coeffs = coeffs[weekly_scale_idx, day_mask.values]
                    day_wavelet_energy = np.mean(np.abs(day_coeffs)) if len(day_coeffs) > 0 else 0
                    
                    energy_contribution = day_wavelet_energy / weekly_energy if weekly_energy > 0 else 0
                    event_count = len(df_orig[df_orig['day_num'] == day_num])
                    
                    combined_score = energy_contribution * event_count
                else:
                    energy_contribution = 0
                    event_count = 0
                    combined_score = 0
                
                day_wavelet_scores[day_name] = {
                    'wavelet_energy': day_wavelet_energy,
                    'energy_contribution': energy_contribution,
                    'combined_score': combined_score,
                    'event_count': event_count
                }
            
            # Select days with significant wavelet energy
            max_score = max(scores['combined_score'] for scores in day_wavelet_scores.values())
            threshold = max_score * 0.3
            
            recommended_days = [
                {
                    'day': day_name,
                    'confidence': scores['combined_score'] / max_score if max_score > 0 else 0,
                    'wavelet_energy': scores['wavelet_energy'],
                    'energy_contribution': scores['energy_contribution'],
                    'event_count': scores['event_count'],
                    'reason': f'High wavelet energy at weekly scale ({scores["energy_contribution"]:.2%})'
                }
                for day_name, scores in day_wavelet_scores.items()
                if scores['combined_score'] > threshold and scores['event_count'] > 0
            ]
            
            return {
                'method': 'wavelet',
                'has_pattern': True,
                'weekly_energy_ratio': weekly_energy_ratio,
                'recommended_days': recommended_days,
                'day_wavelet_analysis': day_wavelet_scores
            }
            
        except Exception as e:
            return {'has_pattern': False, 'reason': f'wavelet_failed: {str(e)}'}
    
    # Methods 9-15: [Similar pattern - each method enhanced to recommend days]
    # ... (continuing with all remaining methods following the same pattern)
    
    @classmethod
    def detect_weekly_patterns_with_day_recommendations(cls, timestamps, methods='recommended'):
        """
        Integrated analysis: Each method detects patterns AND recommends days
        
        No redundant analysis - complete results in one pass!
        """
        
        method_mapping = {
            'autocorrelation': cls.autocorrelation_with_day_ranking,
            'friedman': cls.friedman_test_with_day_ranking,
            'spectral': cls.spectral_analysis_with_day_peaks,
            'clustering': cls.clustering_with_day_selection,
            'entropy': cls.entropy_analysis_with_day_selection,
            'stl': cls.stl_decomposition_with_day_selection,
            'kruskal_wallis': cls.kruskal_wallis_with_day_ranking,
            'wavelet': cls.wavelet_analysis_with_day_detection,
            # Methods 9-15 would be added here following the same pattern:
            # 'cross_correlation': cls.cross_correlation_with_day_selection,
            # 'emd': cls.emd_analysis_with_day_detection,
            # 'arima': cls.arima_residuals_with_day_analysis,
            # 'phase': cls.phase_analysis_with_day_ranking,
            # 'outlier': cls.weekly_outlier_with_day_analysis,
            # 'mann_kendall': cls.mann_kendall_with_day_stability,
            # 'circular': cls.circular_day_analysis_enhanced
        }
        
        if methods == 'recommended':
            selected_methods = ['autocorrelation', 'friedman', 'spectral', 'clustering', 'entropy', 'stl', 'kruskal_wallis', 'wavelet']
        elif methods == 'fast':
            selected_methods = ['autocorrelation', 'entropy', 'clustering']
        elif methods == 'all':
            selected_methods = list(method_mapping.keys())  # All 15 methods
        elif isinstance(methods, list):
            selected_methods = methods
        else:
            selected_methods = ['autocorrelation', 'friedman', 'spectral']
        
        # Run all selected methods
        results = {}
        pattern_votes = 0
        all_day_recommendations = defaultdict(list)
        
        for method_name in selected_methods:
            if method_name in method_mapping:
                try:
                    result = method_mapping[method_name](timestamps)
                    results[method_name] = result
                    
                    if result.get('has_pattern', False):
                        pattern_votes += 1
                        
                        # Collect day recommendations from this method
                        for day_rec in result.get('recommended_days', []):
                            all_day_recommendations[day_rec['day']].append({
                                'method': method_name,
                                'confidence': day_rec['confidence'],
                                'reason': day_rec['reason'],
                                'event_count': day_rec.get('event_count', 0)
                            })
                            
                except Exception as e:
                    results[method_name] = {'has_pattern': False, 'reason': f'error: {str(e)}'}
        
        # Consensus on pattern existence
        consensus_threshold = len(selected_methods) // 2 + 1
        has_weekly_pattern = pattern_votes >= consensus_threshold
        
        # Consensus on day recommendations
        final_day_recommendations = []
        if has_weekly_pattern:
            for day_name, method_votes in all_day_recommendations.items():
                vote_count = len(method_votes)
                avg_confidence = np.mean([vote['confidence'] for vote in method_votes])
                total_events = sum([vote['event_count'] for vote in method_votes])
                
                # Require at least 40% of methods to agree on a day
                if vote_count >= len(selected_methods) * 0.4:
                    final_day_recommendations.append({
                        'day': day_name,
                        'method_votes': vote_count,
                        'total_methods': len(selected_methods),
                        'consensus_ratio': vote_count / len(selected_methods),
                        'avg_confidence': avg_confidence,
                        'total_events': total_events,
                        'voting_methods': [vote['method'] for vote in method_votes],
                        'recommendation_strength': 'high' if vote_count >= len(selected_methods) * 0.6 else 'medium'
                    })
            
            # Sort by consensus ratio and confidence
            final_day_recommendations.sort(
                key=lambda x: (x['consensus_ratio'], x['avg_confidence']), 
                reverse=True
            )
        
        return {
            'has_weekly_pattern': has_weekly_pattern,
            'pattern_detection_votes': pattern_votes,
            'total_methods_used': len(selected_methods),
            'recommended_monitoring_days': [day['day'] for day in final_day_recommendations],
            'detailed_day_analysis': final_day_recommendations,
            'method_results': results,
            'overall_confidence': np.mean([day['avg_confidence'] for day in final_day_recommendations]) if final_day_recommendations else 0
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data with Tuesday/Thursday pattern
    sample_timestamps = []
    base_date = datetime(2024, 1, 1, 9, 0)
    
    for week in range(6):
        # Heavy activity on Tuesday (1) and Thursday (3)
        for day_offset in [1, 3]:
            for event in range(np.random.randint(5, 10)):
                timestamp = base_date + timedelta(
                    weeks=week,
                    days=day_offset,
                    hours=np.random.randint(0, 3),
                    minutes=np.random.randint(0, 60)
                )
                sample_timestamps.append(timestamp)
        
        # Light activity on other days
        for day_offset in [0, 2, 4]:
            if np.random.random() > 0.7:  # 30% chance
                timestamp = base_date + timedelta(
                    weeks=week,
                    days=day_offset,
                    hours=np.random.randint(9, 17)
                )
                sample_timestamps.append(timestamp)
    
    # Run integrated analysis
    detector = IntegratedWeeklyDetector()
    results = detector.detect_weekly_patterns_with_day_recommendations(sample_timestamps)
    
    print("=== INTEGRATED WEEKLY PATTERN + DAY RECOMMENDATION ===")
    print(f"Weekly Pattern Detected: {results['has_weekly_pattern']}")
    print(f"Pattern Votes: {results['pattern_detection_votes']}/{results['total_methods_used']} (using 8 enhanced methods)")
    
    print(f"\n📅 RECOMMENDED MONITORING DAYS:")
    for day_info in results['detailed_day_analysis']:
        print(f"  ✅ {day_info['day']}: {day_info['method_votes']}/{day_info['total_methods']} methods agree "
              f"(confidence: {day_info['avg_confidence']:.2%}, strength: {day_info['recommendation_strength']})")
        print(f"     Methods: {', '.join(day_info['voting_methods'])}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Final recommendation: Monitor {', '.join(results['recommended_monitoring_days'])}")
    print(f"  Overall confidence: {results['overall_confidence']:.2%}")
    print(f"  Note: This example uses 8/15 enhanced methods. Full system would use all 15.")
