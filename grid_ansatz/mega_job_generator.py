#!/usr/bin/env python3
"""
Mega Job 600개 양자 회로 실행 스크립트
- 모든 회로를 한 번의 거대한 job으로 제출
- 최대 효율성과 최소 대기 시간
"""

import os
import sys
import time
import gc
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import simplify_circuit_info, calculate_expressibility_from_ibm_results, calculate_entropy_expressibility_from_ibm_results
from quantum_base import QuantumCircuitBase
from ibm_backend import IBMQuantumBackend
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

def generate_all_circuits():
    """config 설정을 활용한 2큐빗 게이트 비율 테스트용 회로 생성"""
    from config import config
    
    # config에서 기본 설정 가져오기
    circuit_ranges = config.get_circuit_ranges()
    
    # 2큐빗 게이트 비율 테스트 설정
    n_qubits_list = [5, 7 ,10, 15, 20, 60, 127] 
    depth_list = [1, 2, 3, 4]   
    two_qubit_ratios = [0.1, 0.3, 0.5]  # 3개 비율 (10%, 30%, 50%)
    circuits_per_config = 10   # 각 설정당 10개 회로 (할당량 절약)
    
    # 총 회로 수 계산: 3 × 3 × 3 × 10 = 270개
    total_circuits = len(n_qubits_list) * len(depth_list) * len(two_qubit_ratios) * circuits_per_config
    print(f"🔧 테스트용 2큐빗 게이트 비율 테스트 {total_circuits}개 회로 생성 중...")
    print(f"   큐빗 수: {n_qubits_list}")
    print(f"   회로 깊이: {depth_list}")
    print(f"   2큐빗 게이트 비율: {[f'{r:.1%}' for r in two_qubit_ratios]}")
    print(f"   각 설정당 회로 수: {circuits_per_config} (할당량 절약 모드)")
    
    base_circuit = QuantumCircuitBase()
    all_circuits = []
    
    circuit_id = 0
    for n_qubits in n_qubits_list:
        for depth in depth_list:
            for two_qubit_ratio in two_qubit_ratios:
                print(f"  생성 중: {n_qubits}큐빗, 깊이{depth}, 2큐빗비율{two_qubit_ratio:.1%} - {circuits_per_config}개 회로")
                
                for i in range(circuits_per_config):
                    # 회로 생성 (2큐빗 게이트 비율 지정)
                    circuit_info = base_circuit.generate_random_circuit(
                        n_qubits=n_qubits,
                        depth=depth,
                        strategy="hardware_efficient",
                        seed=circuit_id + i,  # 재현 가능한 시드
                        two_qubit_ratio=two_qubit_ratio  # 2큐빗 게이트 비율 설정
                    )
                    
                    # 회로 ID 및 메타데이터 추가
                    circuit_info["circuit_id"] = circuit_id
                    circuit_info["config_group"] = f"q{n_qubits}_d{depth}_r{int(two_qubit_ratio*100)}"
                    circuit_info["two_qubit_ratio_target"] = two_qubit_ratio
                    
                    all_circuits.append(circuit_info)
                    circuit_id += 1
                
                # 진행 상황 출력
                progress = (circuit_id / total_circuits) * 100
                print(f"    진행률: {progress:.1f}% ({circuit_id}/{total_circuits})")
    
    print(f"✅ 총 {len(all_circuits)}개 회로 생성 완료!")
    
    # 설정별 회로 수 요약
    print("\n📊 설정별 회로 수 요약:")
    config_counts = {}
    for circuit in all_circuits:
        config_group = circuit["config_group"]
        if config_group in config_counts:
            config_counts[config_group] = 1
        else:
            config_counts[config_group] = 1
    
    for config_group, count in sorted(config_counts.items()):
        print(f"  {config_group}: {count}개")
    
    return all_circuits

def convert_to_qiskit_circuits(all_circuits, ibm_backend):
    """모든 회로를 Qiskit 회로로 변환"""
    from qiskit import QuantumCircuit, transpile
    
    print("🔄 Qiskit 회로로 변환 및 트랜스파일 중...")
    
    qiskit_circuits = []
    circuit_metadata = []
    
    for i, circuit_data in enumerate(tqdm(all_circuits, desc="회로 변환")):
        try:
            circuit_info = circuit_data['circuit_info']
            n_qubits = circuit_info["n_qubits"]
            gates = circuit_info["gates"]
            wires_list = circuit_info["wires_list"]
            params_idx = circuit_info["params_idx"]
            params = circuit_info["params"]
            
            # 큐빗 수 제한 (백엔드 한계)
            max_backend_qubits = ibm_backend.backend.configuration().n_qubits
            if n_qubits > max_backend_qubits:
                n_qubits = max_backend_qubits
            
            # Qiskit 양자 회로 생성 (U + U†)
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # 순방향 회로 (U) 적용
            for j, (gate, wires) in enumerate(zip(gates, wires_list)):
                if any(w >= n_qubits for w in wires):
                    continue
                    
                if gate == "H":
                    qc.h(wires[0])
                elif gate == "X":
                    qc.x(wires[0])
                elif gate == "Y":
                    qc.y(wires[0])
                elif gate == "Z":
                    qc.z(wires[0])
                elif gate == "S":
                    qc.s(wires[0])
                elif gate == "T":
                    qc.t(wires[0])
                elif gate == "RZ":
                    # 파라미터 찾기
                    param_value = None
                    for k, idx in enumerate(params_idx):
                        if idx == j:
                            param_value = params[k]
                            break
                    if param_value is not None:
                        qc.rz(param_value, wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        qc.cx(wires[0], wires[1])
            
            # 역방향 회로 (U†) 적용
            for j in range(len(gates)-1, -1, -1):
                gate = gates[j]
                wires = wires_list[j]
                
                if any(w >= n_qubits for w in wires):
                    continue
                
                if gate == "H":
                    qc.h(wires[0])
                elif gate == "X":
                    qc.x(wires[0])
                elif gate == "Y":
                    qc.y(wires[0])
                elif gate == "Z":
                    qc.z(wires[0])
                elif gate == "S":
                    qc.sdg(wires[0])
                elif gate == "T":
                    qc.tdg(wires[0])
                elif gate == "RZ":
                    # 파라미터 찾기
                    param_value = None
                    for k, idx in enumerate(params_idx):
                        if idx == j:
                            param_value = params[k]
                            break
                    if param_value is not None:
                        qc.rz(-param_value, wires[0])
                elif gate == "CNOT":
                    if len(wires) >= 2:
                        qc.cx(wires[0], wires[1])
            
            # 측정 추가
            qc.measure_all()
            
            # 트랜스파일
            qc_transpiled = transpile(qc, backend=ibm_backend.backend, optimization_level=0)
            
            # 회로 특성 계산 (트랜스파일 정보 포함)
            circuit_properties = calculate_quantum_properties(circuit_info, qc_transpiled)
            
            # 메타데이터에 회로 특성 추가
            enhanced_metadata = circuit_data.copy()
            enhanced_metadata['circuit_properties'] = circuit_properties
            
            qiskit_circuits.append(qc_transpiled)
            circuit_metadata.append(enhanced_metadata)
            
        except Exception as e:
            print(f"⚠️ 회로 {i} 변환 실패: {str(e)}")
    
    print(f"✅ {len(qiskit_circuits)}개 회로 변환 완료")
    return qiskit_circuits, circuit_metadata

def run_mega_job(qiskit_circuits, circuit_metadata, ibm_backend, shots=128, circuit_shot_requirements=None):
    """
    IBM 백엔드에서 대량 회로를 한 번의 job으로 실행 (진짜 배치 실행)
    
    Args:
        qiskit_circuits (list): Qiskit 회로 목록 (실제로는 circuit_info 목록)
        circuit_metadata (list): 회로 메타데이터 목록
        ibm_backend (IBMQuantumBackend): IBM 백엔드 객체
        shots (int): 기본 회로당 샷 수
        circuit_shot_requirements (list): 각 회로별 필요 샷 수 목록 (선택사항)
        
    Returns:
        list: 실행 결과 목록
    """
    if not qiskit_circuits:
        print("⚠️ 실행할 회로가 없습니다.")
        return []
    
    # 회로별 샷 수 결정
    if circuit_shot_requirements and len(circuit_shot_requirements) == len(qiskit_circuits):
        total_shots = sum(circuit_shot_requirements)
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로별 개별 샷 수: Config 설정에 따라 다름")
        print(f"   배치 총 실행 수: {total_shots:,}")
    else:
        total_shots = len(qiskit_circuits) * shots
        print(f"\n🚀 IBM 백엔드에서 {len(qiskit_circuits)}개 회로를 한 번의 배치 job으로 실행 시작")
        print(f"   회로당 고정 샷 수: {shots:,}")
        print(f"   배치 총 실행 수: {total_shots:,}")
    
    print(f"   예상 데이터 품질: {'🟢 높음' if total_shots/len(qiskit_circuits) >= 1024 else '🟡 보통' if total_shots/len(qiskit_circuits) >= 512 else '🔴 낮음'}")
    
    start_time = time.time()
    
    try:
        # 모든 회로를 Qiskit 회로로 변환
        print("📋 배치 회로 준비 중...")
        qiskit_circuit_list = []
        
        for circuit_idx, circuit_info in enumerate(qiskit_circuits):
            try:
                # circuit_info에서 Qiskit 회로 생성
                n_qubits = circuit_info["n_qubits"]
                gates = circuit_info["gates"]
                wires_list = circuit_info["wires_list"]
                params_idx = circuit_info["params_idx"]
                params = circuit_info["params"]
                
                # 큐빗 수 제한 (백엔드 한계)
                max_backend_qubits = ibm_backend.backend.configuration().n_qubits
                if n_qubits > max_backend_qubits:
                    n_qubits = max_backend_qubits
                
                # Qiskit 양자 회로 생성 (U + U†)
                from qiskit import QuantumCircuit
                qc = QuantumCircuit(n_qubits, n_qubits)
                
                # 순방향 회로 (U) 적용
                for j, (gate, wires) in enumerate(zip(gates, wires_list)):
                    if any(w >= n_qubits for w in wires):
                        continue
                        
                    if gate == "H":
                        qc.h(wires[0])
                    elif gate == "X":
                        qc.x(wires[0])
                    elif gate == "Y":
                        qc.y(wires[0])
                    elif gate == "Z":
                        qc.z(wires[0])
                    elif gate == "S":
                        qc.s(wires[0])
                    elif gate == "T":
                        qc.t(wires[0])
                    elif gate == "RZ":
                        # 파라미터 찾기
                        param_value = None
                        for k, idx in enumerate(params_idx):
                            if idx == j:
                                param_value = params[k]
                                break
                        if param_value is not None:
                            qc.rz(param_value, wires[0])
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])
                
                # 역방향 회로 (U†) 적용
                for j in range(len(gates)-1, -1, -1):
                    gate = gates[j]
                    wires = wires_list[j]
                    
                    if any(w >= n_qubits for w in wires):
                        continue
                    
                    if gate == "H":
                        qc.h(wires[0])
                    elif gate == "X":
                        qc.x(wires[0])
                    elif gate == "Y":
                        qc.y(wires[0])
                    elif gate == "Z":
                        qc.z(wires[0])
                    elif gate == "S":
                        qc.sdg(wires[0])
                    elif gate == "T":
                        qc.tdg(wires[0])
                    elif gate == "RZ":
                        # 파라미터 찾기
                        param_value = None
                        for k, idx in enumerate(params_idx):
                            if idx == j:
                                param_value = params[k]
                                break
                        if param_value is not None:
                            qc.rz(-param_value, wires[0])
                    elif gate == "CNOT":
                        if len(wires) >= 2:
                            qc.cx(wires[0], wires[1])
                
                # 측정 추가
                qc.measure_all()
                
                # 트랜스파일
                from qiskit import transpile
                qc_transpiled = transpile(qc, backend=ibm_backend.backend, optimization_level=0)
                qiskit_circuit_list.append(qc_transpiled)
                
                if (circuit_idx + 1) % 100 == 0:
                    print(f"   회로 변환 진행률: {circuit_idx + 1}/{len(qiskit_circuits)}")
                
            except Exception as e:
                print(f"⚠️ 회로 {circuit_idx} 변환 실패: {str(e)}")
                continue
        
        print(f"✅ {len(qiskit_circuit_list)}개 회로 변환 완료")
        
        if not qiskit_circuit_list:
            print("❌ 변환된 회로가 없습니다.")
            return []
        
        # IBM 백엔드에서 배치 실행
        print("🚀 IBM 백엔드에서 배치 job 제출 중...")
        
        # 각 회로별 샷 수 설정
        if circuit_shot_requirements:
            # 회로별 다른 샷 수 (현재 IBM API는 모든 회로에 동일한 샷 수만 지원)
            # 평균 샷 수 사용
            avg_shots = int(sum(circuit_shot_requirements) / len(circuit_shot_requirements))
            print(f"   회로별 평균 샷 수: {avg_shots}")
        else:
            avg_shots = shots
        
        # IBM Runtime Sampler 사용
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        
        sampler = Sampler(mode=ibm_backend.backend)
        
        # 배치 실행
        print(f"   {len(qiskit_circuit_list)}개 회로를 {avg_shots} 샷으로 실행 중...")
        job = sampler.run(qiskit_circuit_list, shots=avg_shots)
        
        print(f"   Job ID: {job.job_id()}")
        print("   결과 기다리는 중...")
        
        # 결과 대기
        result = job.result()
        
        print("✅ 배치 실행 완료!")
        
        # 결과 처리
        print("📊 결과 처리 중...")
        results = []
        
        for circuit_idx, (pub_result, circuit_info) in enumerate(zip(result, qiskit_circuits)):
            try:
                # 측정 결과 추출
                counts = pub_result.data.meas.get_counts()
                
                # 회로 정보
                n_qubits = circuit_info["n_qubits"]
                
                # 비트열 처리
                processed_counts = {}
                total_counts = sum(counts.values())
                
                for bit_str, count in counts.items():
                    if len(bit_str) > n_qubits:
                        bit_str = bit_str[:n_qubits]
                    elif len(bit_str) < n_qubits:
                        bit_str = bit_str.zfill(n_qubits)
                    
                    if bit_str in processed_counts:
                        processed_counts[bit_str] += count
                    else:
                        processed_counts[bit_str] = count
                
                # 피델리티 계산
                zero_state = '0' * n_qubits
                zero_count = processed_counts.get(zero_state, 0)
                zero_state_probability = zero_count / total_counts if total_counts > 0 else 0
                
                # 오류율 계산
                from main import calculate_error_rates
                error_rates = calculate_error_rates(
                    processed_counts,
                    n_qubits,
                    total_counts
                )
                
                # Robust 피델리티 계산
                from main import calculate_robust_fidelity
                robust_fidelity = calculate_robust_fidelity(
                    processed_counts,
                    n_qubits,
                    total_counts
                )
                
                # 실행 결과 구성
                execution_result = {
                    "zero_state_probability": zero_state_probability,
                    "measurement_counts": processed_counts,
                    "measured_states": total_counts,
                    "significant_states": len(processed_counts),
                    "zero_state_count": zero_count,
                    "error_rates": error_rates,
                    "robust_fidelity": robust_fidelity,
                    "backend": ibm_backend.backend.name
                }
                
                # 표현력 계산 (Classical Shadow + 엔트로피 방법)
                try:
                    # ExpressibilityCalculator 사용
                    from expressibility_calculator import ExpressibilityCalculator, calculate_entropy_expressibility_from_ibm_results
                    calculator = ExpressibilityCalculator()
                    
                    # measurement_counts가 딕셔너리인지 확인
                    if isinstance(processed_counts, dict) and processed_counts:
                        # 1. Classical Shadow 기반 표현력 계산
                        try:
                            # 안전한 Classical Shadow 계산
                            classical_shadow_expressibility = None
                            
                            # 측정 결과가 충분한지 확인
                            if len(processed_counts) > 0 and sum(processed_counts.values()) > 10:
                                # IBM 측정 결과를 Classical Shadow 형태로 변환
                                from main import convert_ibm_results_to_shadow
                                shadow_data = convert_ibm_results_to_shadow(processed_counts, circuit_info["n_qubits"])
                                
                                # Shadow 데이터 유효성 검사
                                if (isinstance(shadow_data, dict) and 
                                    "measurements" in shadow_data and 
                                    "bases" in shadow_data and
                                    isinstance(shadow_data["measurements"], list) and
                                    isinstance(shadow_data["bases"], list) and
                                    len(shadow_data["measurements"]) > 0 and
                                    len(shadow_data["bases"]) > 0):
                                    
                                    shadow_data_list = [shadow_data]
                                    
                                    # ExpressibilityCalculator의 메서드 사용
                                    estimated_moments = calculator._estimate_pauli_expectations_from_shadows(shadow_data_list, circuit_info["n_qubits"])
                                    haar_moments = calculator._get_haar_pauli_expectations(circuit_info["n_qubits"])
                                    
                                    # 거리 계산
                                    distance = calculator._calculate_shadow_distance(estimated_moments, haar_moments)
                                    
                                    classical_shadow_expressibility = {
                                        "n_qubits": circuit_info["n_qubits"],
                                        "samples": 1,
                                        "metric": "classical_shadow",
                                        "distance": distance,
                                        "confidence_interval": [max(0, distance * 0.5), distance * 1.5],
                                        "run_time": 0.1,
                                        "normalized_distance": distance / (2**min(circuit_info["n_qubits"], 10)) if distance > 0 else 0.0,
                                        "source": "ibm_execution_simplified"
                                    }
                                else:
                                    print(f"    ⚠️ 회로 {circuit_idx} Shadow 데이터 구조 오류")
                            else:
                                print(f"    ⚠️ 회로 {circuit_idx} 측정 데이터 부족")
                                
                        except Exception as e:
                            print(f"    ⚠️ 회로 {circuit_idx} Classical Shadow 표현력 계산 오류: {str(e)}")
                            classical_shadow_expressibility = None
                        
                        # 2. 엔트로피 기반 표현력 계산
                        try:
                            entropy_expressibility = calculate_entropy_expressibility_from_ibm_results(
                                processed_counts, circuit_info["n_qubits"]
                            )
                        except Exception as e:
                            print(f"    ⚠️ 회로 {circuit_idx} 엔트로피 표현력 계산 오류: {str(e)}")
                            entropy_expressibility = None
                        
                        # 두 방법의 결과를 모두 저장
                        execution_result["expressibility"] = {
                            "classical_shadow": classical_shadow_expressibility,
                            "entropy_based": entropy_expressibility
                        }
                    else:
                        execution_result["expressibility"] = {
                            "classical_shadow": None,
                            "entropy_based": None
                        }
                except Exception as e:
                    print(f"    ⚠️ 회로 {circuit_idx} 표현력 계산 오류: {str(e)}")
                    execution_result["expressibility"] = {
                        "classical_shadow": None,
                        "entropy_based": None
                    }
                
                # 결과 조합
                result_data = {
                    "circuit_info": circuit_info,
                    "execution_result": execution_result,
                    "circuit_idx": circuit_idx,
                    "shots_used": avg_shots,
                    "execution_time": time.time() - start_time
                }
                
                results.append(result_data)
                
                if (circuit_idx + 1) % 100 == 0:
                    print(f"   결과 처리 진행률: {circuit_idx + 1}/{len(qiskit_circuits)}")
                
            except Exception as e:
                print(f"⚠️ 회로 {circuit_idx} 결과 처리 실패: {str(e)}")
                continue
        
        execution_time = time.time() - start_time
        
        print(f"\n🎉 배치 실행 완료!")
        print(f"   총 실행 시간: {execution_time:.1f}초")
        print(f"   성공한 회로: {len(results)}/{len(qiskit_circuits)}")
        print(f"   성공률: {len(results)/len(qiskit_circuits)*100:.1f}%")
        print(f"   실제 사용된 총 샷 수: {len(results) * avg_shots:,}")
        
        return results
        
    except Exception as e:
        print(f"⚠️ 배치 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_mega_results(result, circuit_metadata, execution_time):
    """mega job 결과 처리 - 실제 측정 데이터만 사용"""
    print("\n📊 결과 처리 중...")
    
    all_results = []
    training_circuits = []
    
    # 표현력 계산을 위한 기본 회로 객체
    base_circuit = QuantumCircuitBase()
    
    for i, (circuit_result, circuit_data) in enumerate(zip(result, circuit_metadata)):
        try:
            circuit_info = circuit_data['circuit_info']
            n_qubits = circuit_info["n_qubits"]
            
            # 측정 결과 추출
            counts = {}
            
            # Qiskit 2.0+ API에서 결과 추출
            if hasattr(circuit_result, 'data') and hasattr(circuit_result.data, 'meas'):
                bit_array = circuit_result.data.meas
                if hasattr(bit_array, 'get_counts'):
                    counts_dict = bit_array.get_counts()
                    
                    # 비트열 처리
                    sparse_counts = {}
                    for bit_str, count in counts_dict.items():
                        if len(bit_str) > n_qubits:
                            bit_str = bit_str[:n_qubits]
                        elif len(bit_str) < n_qubits:
                            bit_str = bit_str.zfill(n_qubits)
                        
                        if bit_str in sparse_counts:
                            sparse_counts[bit_str] += count
                        else:
                            sparse_counts[bit_str] = count
                    
                    counts = sparse_counts
            
            if not counts:
                print(f"⚠️ 회로 {i} 결과 추출 실패")
                continue
            
            # 실제 측정 결과 기반 통계
            measurement_stats = calculate_measurement_statistics(counts, n_qubits)
            
            # 피델리티 계산 (실제 측정 결과)
            total_counts = sum(counts.values())
            zero_state = '0' * n_qubits
            zero_count = counts.get(zero_state, 0)
            fidelity = zero_count / total_counts if total_counts > 0 else 0
            
            # Robust Fidelity 계산 (노이즈 허용)
            robust_fidelity = calculate_robust_fidelity_mega(counts, n_qubits, total_counts)
            
            # 표현력 계산 (IBM 측정 결과 기반)
            if i % 50 == 0:
                print(f"  회로 {i+1}/{len(result)} 표현력 계산 중...")
            
            try:
                # 1. Classical Shadow 기반 표현력 계산
                expressibility_result = calculate_expressibility_from_ibm_results(
                    base_circuit, 
                    circuit_info,
                    counts,  # 측정 결과 사용
                    n_qubits,
                    samples=1  # 단일 측정 기반
                )
                
                # 표현력 관련 지표 추출
                expr_distance = expressibility_result.get("distance", 0)
                expr_normalized = expressibility_result.get("normalized_distance", 0)
                
            except Exception as e:
                print(f"⚠️ 회로 {i} Classical Shadow 표현력 계산 실패: {str(e)}")
                expressibility_result = None
                expr_distance = 0
                expr_normalized = 0
            
            # 2. 엔트로피 기반 표현력 계산 (새로운 방식)
            try:
                # QuantumCircuitBase를 사용하여 엔트로피 계산
                entropy_value = base_circuit.calculate_entropy(counts)
                
                # 결과 포맷팅 (이전 버전과의 호환성을 위해 유사한 구조 유지)
                entropy_expressibility_result = {
                    "expressibility_value": entropy_value,
                    "entropy": entropy_value,
                    "method": "measurement_entropy"
                }
                
                # 이전 버전과의 호환성을 위한 별칭
                entropy_expr_value = entropy_value
                
            except Exception as e:
                print(f"⚠️ 회로 {i} 엔트로피 계산 실패: {str(e)}")
                entropy_expressibility_result = None
                entropy_expr_value = 0
                angle_entropy = 0
                distance_entropy = 0
            
            # 회로 구조 특성 가져오기
            circuit_properties = circuit_data.get('circuit_properties', {})
            structural_props = circuit_properties.get('structural_properties', {})
            param_props = circuit_properties.get('parameter_properties', {})
            hardware_props = circuit_properties.get('hardware_context', {})
            
            # 트랜스포머 입력용 임베딩 생성
            circuit_sequence = circuit_properties.get('circuit_sequence', [])
            circuit_embedding = create_circuit_embedding(circuit_sequence)
            
            # AI 훈련용 데이터 구조 (실제 데이터만)
            training_circuit = {
                "circuit_id": circuit_data['circuit_id'],
                "ansatz_data": {
                    "circuit_info": simplify_circuit_info(circuit_info),
                    "execution_results": {
                        "fidelity": fidelity,
                        "robust_fidelity": robust_fidelity,
                        "measured_states": total_counts,
                        "significant_states": len(counts),
                        "zero_state_count": zero_count,
                        "backend": "ibm_mega_job"
                    },
                    "expressibility": expressibility_result,
                    "entropy_expressibility": entropy_expressibility_result,
                    "metadata": circuit_data['metadata']
                },
                "features": {
                    # 기본 회로 정보
                    "n_qubits": circuit_data['n_qubits'],
                    "depth": circuit_data['depth'],
                    "gate_count": len(circuit_info.get("gates", [])),
                    
                    # 실제 측정 결과
                    "fidelity": fidelity,
                    "robust_fidelity": robust_fidelity,
                    "expressibility_distance": expr_distance,
                    "normalized_expressibility": expr_normalized,
                        
                    # 엔트로피 기반 표현력 지표
                    "entropy_expressibility": entropy_expr_value,
                    "angle_entropy": angle_entropy,
                    "distance_entropy": distance_entropy,
                        
                    # 측정 통계 (실제 데이터)
                    **measurement_stats,
                    
                    # 회로 구조 특성 (실제 데이터)
                    **structural_props,
                    
                    # 파라미터 특성 (실제 데이터)
                    **param_props,
                    
                    # 하드웨어 최적화 결과 (실제 데이터)
                    **hardware_props,
                    
                    # 시퀀스 특성 (트랜스포머 친화적)
                    **circuit_embedding.get("sequence_features", {}),
                    
                    # 실행 정보
                    "execution_time": execution_time / len(circuit_metadata)
                },
                "transformer_input": {
                    # 트랜스포머 입력용 시퀀스
                    "gate_sequence": circuit_embedding.get("gate_sequence", []),
                    "qubit_sequence": circuit_embedding.get("qubit_sequence", []),
                    "param_sequence": circuit_embedding.get("param_sequence", []),
                    "gate_type_sequence": circuit_embedding.get("gate_type_sequence", [])
                },
                "labels": {
                    "complexity_class": "low" if circuit_data['n_qubits'] <= 50 else "medium" if circuit_data['n_qubits'] <= 90 else "high",
                    "depth_class": "shallow" if circuit_data['depth'] <= 2 else "deep",
                    "fidelity_class": "high" if fidelity > 0.1 else "medium" if fidelity > 0.01 else "low",
                    "robust_fidelity_class": "high" if robust_fidelity > 0.15 else "medium" if robust_fidelity > 0.05 else "low",
                    "expressibility_class": "high" if expr_normalized < 0.001 else "medium" if expr_normalized < 0.01 else "low",
                    "entropy_expressibility_class": "high" if entropy_expr_value > 4.0 else "medium" if entropy_expr_value > 2.0 else "low",
                    "angle_entropy_class": "high" if angle_entropy > 2.0 else "medium" if angle_entropy > 1.0 else "low",
                    "distance_entropy_class": "high" if distance_entropy > 2.0 else "medium" if distance_entropy > 1.0 else "low"
                }
            }
            
            training_circuits.append(training_circuit)
            
            # CSV용 결과 (플랫 구조)
            circuit_result_data = {
                "circuit_id": circuit_data['circuit_id'],
                "n_qubits": circuit_data['n_qubits'],
                "depth": circuit_data['depth'],
                "gate_count": len(circuit_info.get("gates", [])),
                "fidelity": fidelity,
                "robust_fidelity": robust_fidelity,
                "expressibility_distance": expr_distance,
                "normalized_expressibility": expr_normalized,
                "execution_time": execution_time / len(circuit_metadata),
                "backend": "ibm_mega_job",
                
                # 엔트로피 기반 표현력 지표
                "entropy_expressibility": entropy_expr_value,
                "angle_entropy": angle_entropy,
                "distance_entropy": distance_entropy,
                
                # 측정 통계
                **measurement_stats,
                
                # 구조 특성 (커플링 특성 포함)
                **{f"struct_{k}": v for k, v in structural_props.items()},
                
                # 파라미터 특성
                **{f"param_{k}": v for k, v in param_props.items()},
                
                # 하드웨어 특성
                **{f"hw_{k}": v for k, v in hardware_props.items()},
                
                # 시퀀스 특성 (트랜스포머 친화적)
                **{f"seq_{k}": v for k, v in circuit_embedding.get("sequence_features", {}).items()}
            }
            
            all_results.append(circuit_result_data)
            
        except Exception as e:
            print(f"⚠️ 회로 {i} 처리 실패: {str(e)}")
    
    print(f"\n✅ {len(all_results)}개 회로 결과 처리 완료")
    return all_results, training_circuits

def save_mega_results(all_results, training_circuits):
    """mega job 결과 저장"""
    print("\n💾 결과 저장 중...")
    
    # 디렉토리 생성
    results_dir = "grid_circuits/mega_results"
    training_data_dir = "grid_circuits/training_data"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)
    
    # 테스트 ID 생성
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_file = os.path.join(results_dir, f"mega_job_results_{test_id}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"CSV 결과 저장: {csv_file}")
        print(f"  총 컬럼 수: {len(results_df.columns)}")
        print(f"  주요 컬럼: fidelity, normalized_expressibility, entropy, concentration, cnot_count, etc.")
    
    # AI 훈련용 배치 저장 (100개씩)
    circuits_per_batch = 100
    batch_counter = 0
    
    for i in range(0, len(training_circuits), circuits_per_batch):
        batch_circuits = training_circuits[i:i+circuits_per_batch]
        
        # 배치 통계 계산
        fidelities = [c["features"]["fidelity"] for c in batch_circuits]
        robust_fidelities = [c["features"]["robust_fidelity"] for c in batch_circuits]
        expressibilities = [c["features"]["normalized_expressibility"] for c in batch_circuits]
        entropies = [c["features"]["entropy"] for c in batch_circuits]
        concentrations = [c["features"]["concentration"] for c in batch_circuits]
        
        batch_metadata = {
            "batch_id": batch_counter,
            "test_id": test_id,
            "backend": "ibm_mega_job",
            "batch_created": datetime.now().isoformat(),
            "circuits_per_batch": len(batch_circuits),
            "is_mega_job": True,
            "data_type": "real_measurement_only"
        }
        
        training_batch = {
            "metadata": batch_metadata,
            "circuits": batch_circuits,
            "statistics": {
                "total_circuits": len(batch_circuits),
                "fidelity_stats": {
                    "mean": np.mean(fidelities),
                    "std": np.std(fidelities),
                    "min": np.min(fidelities),
                    "max": np.max(fidelities)
                },
                "robust_fidelity_stats": {
                    "mean": np.mean(robust_fidelities),
                    "std": np.std(robust_fidelities),
                    "min": np.min(robust_fidelities),
                    "max": np.max(robust_fidelities)
                },
                "expressibility_stats": {
                    "mean": np.mean(expressibilities),
                    "std": np.std(expressibilities),
                    "min": np.min(expressibilities),
                    "max": np.max(expressibilities)
                },
                "entropy_stats": {
                    "mean": np.mean(entropies),
                    "std": np.std(entropies),
                    "min": np.min(entropies),
                    "max": np.max(entropies)
                },
                "concentration_stats": {
                    "mean": np.mean(concentrations),
                    "std": np.std(concentrations),
                    "min": np.min(concentrations),
                    "max": np.max(concentrations)
                }
            }
        }
        
        # 배치 파일 저장
        batch_filename = os.path.join(training_data_dir, f"mega_batch_{batch_counter:03d}_{test_id}.json")
        with open(batch_filename, 'w') as f:
            json.dump(training_batch, f, indent=2, default=str)
        
        # 압축 파일 저장
        import gzip
        compressed_filename = batch_filename.replace('.json', '.json.gz')
        with gzip.open(compressed_filename, 'wt') as f:
            json.dump(training_batch, f, indent=1, default=str)
        
        print(f"배치 {batch_counter} 저장: {len(batch_circuits)}개 회로")
        print(f"  피델리티: {np.mean(fidelities):.4f}±{np.std(fidelities):.4f}")
        print(f"  Robust 피델리티: {np.mean(robust_fidelities):.4f}±{np.std(robust_fidelities):.4f}")
        print(f"  표현력: {np.mean(expressibilities):.4f}±{np.std(expressibilities):.4f}")
        print(f"  엔트로피: {np.mean(entropies):.2f}±{np.std(entropies):.2f}")
        batch_counter += 1
    
    return results_df if all_results else pd.DataFrame()

def calculate_quantum_properties(circuit_info, qc_transpiled=None):
    """실제 회로 구조에서 나오는 수치적 특성만 계산"""
    n_qubits = circuit_info["n_qubits"]
    gates = circuit_info["gates"]
    wires_list = circuit_info["wires_list"]
    params = circuit_info.get("params", [])
    
    # 1. 회로 시퀀스 (트랜스포머 입력용) - 실제 게이트 순서
    circuit_sequence = []
    for i, (gate, wires) in enumerate(zip(gates, wires_list)):
        gate_info = {
            "gate": gate,
            "qubits": wires,
            "params": []
        }
        
        # 실제 파라미터 값
        if i in circuit_info.get("params_idx", []):
            param_idx = circuit_info["params_idx"].index(i)
            if param_idx < len(params):
                gate_info["params"] = [params[param_idx]]
        
        circuit_sequence.append(gate_info)
    
    # 2. 실제 게이트 통계
    gate_counts = {}
    for gate in gates:
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    
    # 3. 실제 연결성 (CNOT 게이트 기반)
    cnot_connections = set()
    for gate, wires in zip(gates, wires_list):
        if gate == "CNOT" and len(wires) >= 2:
            cnot_connections.add(tuple(sorted(wires[:2])))
    
    # 4. 커플링맵을 트랜스포머 친화적 형태로 변환
    coupling_map = circuit_info.get("coupling_map", [])
    coupling_features = create_coupling_features(coupling_map, n_qubits)
    
    # 5. 실제 파라미터 통계
    param_stats = {}
    if params:
        param_stats = {
            "param_count": len(params),
            "param_mean": np.mean(params),
            "param_std": np.std(params),
            "param_min": np.min(params),
            "param_max": np.max(params)
        }
    else:
        param_stats = {
            "param_count": 0,
            "param_mean": 0,
            "param_std": 0,
            "param_min": 0,
            "param_max": 0
        }
    
    # 6. 하드웨어 최적화 결과 (실제 트랜스파일 결과)
    hardware_context = {}
    if qc_transpiled:
        original_gate_count = len(gates)
        transpiled_gate_count = len(qc_transpiled.data)
        
        hardware_context = {
            "original_depth": circuit_info["depth"],
            "transpiled_depth": qc_transpiled.depth(),
            "original_gate_count": original_gate_count,
            "transpiled_gate_count": transpiled_gate_count,
            "gate_overhead": transpiled_gate_count / original_gate_count if original_gate_count > 0 else 1.0,
            "depth_overhead": qc_transpiled.depth() / circuit_info["depth"] if circuit_info["depth"] > 0 else 1.0
        }
    
    return {
        "circuit_sequence": circuit_sequence,
        "structural_properties": {
            "n_qubits": n_qubits,
            "depth": circuit_info["depth"],
            "total_gates": len(gates),
            "cnot_count": gate_counts.get("CNOT", 0),
            "single_qubit_gates": sum(gate_counts.get(g, 0) for g in ["H", "X", "Y", "Z", "S", "T", "RZ"]),
            "unique_gate_types": len(gate_counts),
            "cnot_connections": len(cnot_connections),
            "gate_counts": gate_counts,
            **coupling_features  # 커플링 특성 추가
        },
        "parameter_properties": param_stats,
        "hardware_context": hardware_context
    }

def create_coupling_features(coupling_map, n_qubits):
    """커플링맵을 트랜스포머가 이해하기 쉬운 수치적 특성으로 변환"""
    if not coupling_map:
        return {
            "coupling_density": 0.0,
            "max_degree": 0,
            "avg_degree": 0.0,
            "connectivity_ratio": 0.0,
            "diameter": 0,
            "clustering_coefficient": 0.0
        }
    
    # 인접 리스트 생성
    adjacency = {i: set() for i in range(n_qubits)}
    for edge in coupling_map:
        if len(edge) >= 2:
            a, b = edge[0], edge[1]
            if a < n_qubits and b < n_qubits:
                adjacency[a].add(b)
                adjacency[b].add(a)
    
    # 1. 커플링 밀도 (실제 연결 / 최대 가능 연결)
    total_edges = len(coupling_map)
    max_edges = n_qubits * (n_qubits - 1) // 2
    coupling_density = total_edges / max_edges if max_edges > 0 else 0
    
    # 2. 차수 통계
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    max_degree = max(degrees) if degrees else 0
    avg_degree = np.mean(degrees) if degrees else 0
    
    # 3. 연결성 비율 (연결된 큐빗 / 전체 큐빗)
    connected_qubits = sum(1 for d in degrees if d > 0)
    connectivity_ratio = connected_qubits / n_qubits if n_qubits > 0 else 0
    
    # 4. 그래프 지름 (최단 경로의 최대값) - 간단한 BFS
    diameter = calculate_graph_diameter(adjacency, n_qubits)
    
    # 5. 클러스터링 계수 (삼각형 형성 정도)
    clustering_coefficient = calculate_clustering_coefficient(adjacency)
    
    return {
        "coupling_density": coupling_density,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "connectivity_ratio": connectivity_ratio,
        "diameter": diameter,
        "clustering_coefficient": clustering_coefficient
    }

def calculate_graph_diameter(adjacency, n_qubits):
    """그래프의 지름 계산 (BFS 기반)"""
    from collections import deque
    
    max_distance = 0
    
    for start in range(n_qubits):
        if not adjacency[start]:  # 연결되지 않은 노드는 건너뜀
            continue
            
        # BFS로 최단 거리 계산
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
                    max_distance = max(max_distance, distances[neighbor])
    
    return max_distance

def calculate_clustering_coefficient(adjacency):
    """클러스터링 계수 계산"""
    total_coefficient = 0
    valid_nodes = 0
    
    for node, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
            
        # 이웃 노드들 간의 연결 수 계산
        neighbor_list = list(neighbors)
        possible_edges = len(neighbor_list) * (len(neighbor_list) - 1) // 2
        actual_edges = 0
        
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adjacency[neighbor_list[i]]:
                    actual_edges += 1
        
        if possible_edges > 0:
            total_coefficient += actual_edges / possible_edges
            valid_nodes += 1
    
    return total_coefficient / valid_nodes if valid_nodes > 0 else 0

def create_circuit_embedding(circuit_sequence, max_length=100):
    """회로 시퀀스를 트랜스포머 입력용 임베딩으로 변환 - 개선된 버전"""
    # 게이트 타입을 숫자로 매핑
    gate_to_id = {
        "H": 1, "X": 2, "Y": 3, "Z": 4, "S": 5, "T": 6, "RZ": 7, "CNOT": 8,
        "PAD": 0  # 패딩용
    }
    
    # 시퀀스를 고정 길이로 변환
    gate_sequence = []
    qubit_sequence = []
    param_sequence = []
    gate_type_sequence = []  # 단일/이중 큐빗 게이트 구분
    
    for gate_info in circuit_sequence[:max_length]:
        gate_id = gate_to_id.get(gate_info["gate"], 0)
        gate_sequence.append(gate_id)
        
        # 큐빗 위치 정보 (첫 번째 큐빗만 사용, 패딩은 -1)
        if gate_info["qubits"]:
            qubit_sequence.append(gate_info["qubits"][0])
        else:
            qubit_sequence.append(-1)
        
        # 파라미터 값 (없으면 0)
        if gate_info["params"]:
            param_sequence.append(gate_info["params"][0])
        else:
            param_sequence.append(0.0)
        
        # 게이트 타입 (1: 단일 큐빗, 2: 이중 큐빗)
        if gate_info["gate"] == "CNOT":
            gate_type_sequence.append(2)
        else:
            gate_type_sequence.append(1)
    
    # 패딩
    while len(gate_sequence) < max_length:
        gate_sequence.append(0)
        qubit_sequence.append(-1)
        param_sequence.append(0.0)
        gate_type_sequence.append(0)
    
    # 추가 시퀀스 특성 계산
    sequence_features = {
        "sequence_length": min(len(circuit_sequence), max_length),
        "unique_gates_in_sequence": len(set(gate_sequence[:min(len(circuit_sequence), max_length)])),
        "param_gate_ratio": sum(1 for p in param_sequence if p != 0) / max_length,
        "two_qubit_gate_ratio": sum(1 for t in gate_type_sequence if t == 2) / max_length
    }
    
    return {
        "gate_sequence": gate_sequence,
        "qubit_sequence": qubit_sequence,
        "param_sequence": param_sequence,
        "gate_type_sequence": gate_type_sequence,
        "sequence_features": sequence_features
    }

def calculate_measurement_statistics(counts, n_qubits):
    """실제 측정 결과에서만 나오는 통계적 특성"""
    if not counts:
        return {}
    
    total_counts = sum(counts.values())
    probabilities = np.array([count / total_counts for count in counts.values()])
    
    # 1. 실제 측정 분포 특성
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # 2. 실제 상태 분포
    zero_state = '0' * n_qubits
    zero_prob = counts.get(zero_state, 0) / total_counts
    
    # 3. 측정 집중도 (가장 많이 측정된 상태의 비율)
    max_count = max(counts.values())
    concentration = max_count / total_counts
    
    # 4. 유효 상태 수 (실제로 측정된 상태 수)
    measured_states = len(counts)
    
    # 5. 분산 및 표준편차
    count_values = list(counts.values())
    measurement_variance = np.var(count_values)
    measurement_std = np.std(count_values)
    
    # 6. 상위 상태들의 누적 확률
    sorted_probs = sorted(probabilities, reverse=True)
    top_1_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0
    top_5_prob = sum(sorted_probs[:5]) if len(sorted_probs) >= 5 else sum(sorted_probs)
    top_10_prob = sum(sorted_probs[:10]) if len(sorted_probs) >= 10 else sum(sorted_probs)
    
    return {
        "entropy": entropy,
        "zero_state_probability": zero_prob,
        "concentration": concentration,
        "measured_states": measured_states,
        "measurement_variance": measurement_variance,
        "measurement_std": measurement_std,
        "top_1_probability": top_1_prob,
        "top_5_probability": top_5_prob,
        "top_10_probability": top_10_prob,
        "total_measurements": total_counts
    }

def calculate_robust_fidelity_mega(counts, n_qubits, total_counts):
    """
    메가잡용 Robust Fidelity 계산 (노이즈 허용)
    
    Args:
        counts (dict): 측정 결과 카운트 {비트열: 카운트}
        n_qubits (int): 큐빗 수
        total_counts (int): 총 측정 횟수
        
    Returns:
        float: Robust Fidelity (0~1 사이)
    """
    if total_counts == 0:
        return 0.0
    
    # 목표 상태 (모든 비트가 0)
    target_state = '0' * n_qubits
    
    # 허용 오류 비트 수
    error_threshold = get_error_threshold_mega(n_qubits)
    
    # 허용 범위 내의 모든 측정 카운트 합산
    robust_count = 0
    
    for measured_state, count in counts.items():
        # 측정된 상태와 목표 상태 간의 해밍 거리 계산
        distance = hamming_distance_mega(measured_state, target_state)
        
        # 허용 범위 내이면 카운트에 포함
        if distance <= error_threshold:
            robust_count += count
    
    # Robust Fidelity 계산
    robust_fidelity = robust_count / total_counts
    
    return robust_fidelity

def get_error_threshold_mega(n_qubits):
    """
    메가잡용 큐빗 수에 따른 허용 오류 비트 수 계산
    
    Args:
        n_qubits (int): 큐빗 수
        
    Returns:
        int: 허용 오류 비트 수
    """
    if n_qubits <= 10:
        return 1  # 10큐빗 이하는 1개 오류만 허용
    else:
        return max(1, int(n_qubits * 0.1))  # 10% 이내 오류 허용

def hamming_distance_mega(state1, state2):
    """
    메가잡용 두 비트 문자열 간의 해밍 거리 계산
    
    Args:
        state1 (str): 첫 번째 비트 문자열
        state2 (str): 두 번째 비트 문자열
        
    Returns:
        int: 해밍 거리 (다른 비트 수)
    """
    if len(state1) != len(state2):
        return float('inf')  # 길이가 다르면 무한대 거리
    
    return sum(c1 != c2 for c1, c2 in zip(state1, state2))

def calculate_optimal_shots_and_batching(total_circuits, target_total_shots=8000000, max_executions=10000000):
    """
    Config 설정 기반 최적 샷 수 및 배치 분할 계산
    
    Args:
        total_circuits (int): 총 회로 수
        target_total_shots (int): 목표 총 샷 수 (참고용)
        max_executions (int): IBM 제한 최대 실행 수
        
    Returns:
        dict: 배치 분할 정보
    """
    from config import config
    
    print("\n🎯 Config 기반 최적 샷 수 및 배치 분할 계산")
    
    # 회로별 실제 필요 샷 수 계산 (config 설정 기반)
    # 예시 회로들의 큐빗 수 분포 (5, 7, 10 큐빗)
    circuit_shot_requirements = []
    
    # 각 회로 설정별 샷 수 계산
    qubit_configs = [5, 7, 10]  # 실제 생성되는 회로의 큐빗 수
    circuits_per_config = total_circuits // (3 * 3 * 3)  # 3×3×3 = 27개 설정
    
    for n_qubits in qubit_configs:
        # 피델리티 측정용 샷 수 (config 기반)
        fidelity_shots = config.get_ibm_shots(n_qubits)
        
        # 표현력 측정용 샷 수 (config 기반)
        expressibility_samples = config.ibm_backend.expressibility_samples  # 32
        expressibility_shots_per_sample = config.ibm_backend.expressibility_shots  # 64
        expressibility_total_shots = expressibility_samples * expressibility_shots_per_sample  # 32 × 64 = 2,048
        
        # 회로당 총 샷 수
        shots_per_circuit = fidelity_shots + expressibility_total_shots
        
        print(f"  {n_qubits}큐빗 회로:")
        print(f"    피델리티 측정: {fidelity_shots} 샷")
        print(f"    표현력 측정: {expressibility_samples} 샘플 × {expressibility_shots_per_sample} 샷 = {expressibility_total_shots} 샷")
        print(f"    회로당 총 샷 수: {shots_per_circuit:,} 샷")
        
        # 해당 큐빗 수의 회로 수만큼 추가
        num_circuits_this_config = circuits_per_config * 9  # 3개 깊이 × 3개 비율
        for _ in range(num_circuits_this_config):
            circuit_shot_requirements.append(shots_per_circuit)
    
    # 전체 필요 샷 수 계산
    total_required_shots = sum(circuit_shot_requirements)
    
    print(f"\n📊 전체 샷 수 요구사항:")
    print(f"  총 회로 수: {total_circuits:,}")
    print(f"  총 필요 샷 수: {total_required_shots:,}")
    print(f"  목표 샷 수: {target_total_shots:,}")
    print(f"  IBM 제한: {max_executions:,} 실행")
    
    # 배치 분할 계산 (IBM 제한 내에서)
    batches = []
    current_batch_shots = 0
    current_batch_circuits = []
    
    for i, shots_needed in enumerate(circuit_shot_requirements):
        # 현재 배치에 추가했을 때 IBM 제한을 초과하는지 확인
        if current_batch_shots + shots_needed > max_executions:
            # 현재 배치 완료하고 새 배치 시작
            if current_batch_circuits:
                batches.append({
                    "circuits": current_batch_circuits.copy(),
                    "total_shots": current_batch_shots,
                    "circuit_count": len(current_batch_circuits)
                })
                print(f"  배치 {len(batches)}: {len(current_batch_circuits)}개 회로, {current_batch_shots:,} 샷")
            
            # 새 배치 시작
            current_batch_circuits = [i]
            current_batch_shots = shots_needed
        else:
            # 현재 배치에 추가
            current_batch_circuits.append(i)
            current_batch_shots += shots_needed
    
    # 마지막 배치 추가
    if current_batch_circuits:
        batches.append({
            "circuits": current_batch_circuits.copy(),
            "total_shots": current_batch_shots,
            "circuit_count": len(current_batch_circuits)
        })
        print(f"  배치 {len(batches)}: {len(current_batch_circuits)}개 회로, {current_batch_shots:,} 샷")
    
    # 배치 정보 요약
    total_batches = len(batches)
    circuits_per_batch = [batch["circuit_count"] for batch in batches]
    executions_per_batch = [batch["total_shots"] for batch in batches]
    
    print(f"\n✅ 배치 분할 완료:")
    print(f"  총 배치 수: {total_batches}")
    print(f"  배치별 회로 수: {circuits_per_batch}")
    print(f"  배치별 실행 수: {[f'{shots:,}' for shots in executions_per_batch]}")
    print(f"  총 실행 수: {sum(executions_per_batch):,}")
    print(f"  IBM 제한 준수: {'✅' if all(shots <= max_executions for shots in executions_per_batch) else '❌'}")
    
    # 샷 수 적절성 검증
    print(f"\n🔍 샷 수 적절성 검증:")
    avg_shots_per_circuit = total_required_shots / total_circuits
    print(f"   회로당 평균 샷 수: {avg_shots_per_circuit:,.0f}")
    
    # 피델리티 측정 신뢰성 (최소 1024샷 권장)
    min_fidelity_shots = min([config.get_ibm_shots(n) for n in qubit_configs])
    fidelity_adequate = min_fidelity_shots >= 1024
    print(f"   피델리티 측정 신뢰성: {'✅ 충분' if fidelity_adequate else '⚠️ 부족'} ({min_fidelity_shots} ≥ 1024)")
    
    # 표현력 계산 정확성 (최소 512샷 권장)
    expressibility_shots = config.ibm_backend.expressibility_samples * config.ibm_backend.expressibility_shots
    expressibility_adequate = expressibility_shots >= 512
    print(f"   표현력 계산 정확성: {'✅ 적절' if expressibility_adequate else '⚠️ 부족'} ({expressibility_shots} ≥ 512)")
    
    # 통계적 유의성 (최소 256샷 권장)
    statistical_adequate = avg_shots_per_circuit >= 256
    print(f"   통계적 유의성: {'✅ 보장' if statistical_adequate else '⚠️ 부족'} ({avg_shots_per_circuit:.0f} ≥ 256)")
    
    return {
        "num_batches": total_batches,
        "circuits_per_batch": circuits_per_batch,
        "executions_per_batch": executions_per_batch,
        "total_executions": sum(executions_per_batch),
        "circuit_shot_requirements": circuit_shot_requirements,
        "batches": batches,
        "data_integrity": "완전 보장 (Config 기반 적정 샷 수)",
        "shots_adequacy": {
            "fidelity_adequate": fidelity_adequate,
            "expressibility_adequate": expressibility_adequate,
            "statistical_adequate": statistical_adequate
        }
    }

def setup_directories():
    """프로그램 실행에 필요한 모든 디렉토리 생성"""
    print("📁 필요한 디렉토리들을 생성 중...")
    
    # 생성할 디렉토리 목록
    directories = [
        "grid_circuits",                    # 메인 디렉토리
        "grid_circuits/mega_results",       # 메가잡 결과 저장
        "grid_circuits/training_data",      # AI 훈련 데이터
        "grid_circuits/analysis",           # 분석 결과
        "grid_circuits/logs",               # 로그 파일
        "grid_circuits/temp",               # 임시 파일
        "grid_circuits/backup",             # 백업 파일
        "models",                           # 훈련된 모델 저장
        "plots",                            # 시각화 결과
        "reports"                           # 보고서
    ]
    
    created_dirs = []
    existing_dirs = []
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                created_dirs.append(directory)
                print(f"  ✅ 생성됨: {directory}")
            else:
                existing_dirs.append(directory)
                print(f"  📂 이미 존재: {directory}")
        except Exception as e:
            print(f"  ❌ 생성 실패: {directory} - {str(e)}")
    
    print(f"\n📊 디렉토리 설정 완료:")
    print(f"  새로 생성: {len(created_dirs)}개")
    print(f"  기존 존재: {len(existing_dirs)}개")
    
    # 권한 확인
    test_file_path = os.path.join("grid_circuits/mega_results", "test_write.tmp")
    try:
        with open(test_file_path, 'w') as f:
            f.write("test")
        os.remove(test_file_path)
        print(f"  ✅ 쓰기 권한 확인 완료")
    except Exception as e:
        print(f"  ⚠️ 쓰기 권한 문제: {str(e)}")
    
    return True

def run_mega_job_generator():
    """메인 2큐빗 게이트 비율 테스트 실행 함수 (1800개 회로)"""
    
    start_time = time.time()
    
    try:
        # 0️⃣ 디렉토리 설정
        print("0️⃣ 프로그램 환경 설정")
        setup_directories()
        
        # IBM 백엔드 초기화
        print("\n1️⃣ IBM Quantum 백엔드 초기화")
        ibm_token = os.environ.get('IBM_QUANTUM_TOKEN')
        if not ibm_token:
            print("❌ IBM_QUANTUM_TOKEN 환경 변수가 필요합니다.")
            return
        
        ibm_backend = IBMQuantumBackend(ibm_token=ibm_token)
        if not ibm_backend or not ibm_backend.backend:
            print("❌ IBM 백엔드 연결 실패")
            return
        
        print(f"✅ {ibm_backend.backend.name} 연결 성공")
        
        # 2큐빗 게이트 비율 테스트용 회로 생성
        print("\n2️⃣ 2큐빗 게이트 비율 테스트용 1800개 회로 생성")
        all_circuits = generate_all_circuits()
        
        # Qiskit 회로로 변환
        print("\n3️⃣ Qiskit 회로 변환 및 검증")
        qiskit_circuits = []
        conversion_errors = 0
        
        print(f"1800개 회로를 Qiskit 형식으로 변환 중...")
        for i, circuit_info in enumerate(all_circuits):
            try:
                # 회로 정보에서 직접 사용 (이미 circuit_info 형태)
                qiskit_circuits.append(circuit_info)
                
                # 진행 상황 출력 (100개마다)
                if (i + 1) % 100 == 0:
                    print(f"  변환 진행률: {(i+1)/len(all_circuits)*100:.1f}% ({i+1}/{len(all_circuits)})")
                    
            except Exception as e:
                print(f"⚠️ 회로 {i} 변환 오류: {str(e)}")
                conversion_errors += 1
        
        print(f"✅ {len(qiskit_circuits)}개 회로 변환 완료 (오류: {conversion_errors}개)")
        
        # 배치 처리로 IBM 백엔드 실행
        print("\n4️⃣ IBM 백엔드에서 최적화된 배치 실행")
        
        # 800만 샷 내외로 최적화된 배치 계산
        target_total_shots = 8000000  # 800만 샷 목표
        
        print(f"목표 총 샷 수: {target_total_shots:,}")
        
        # 최적 샷 수 및 배치 분할 계산
        batch_info = calculate_optimal_shots_and_batching(
            total_circuits=len(qiskit_circuits),
            target_total_shots=target_total_shots,
            max_executions=10000000  # IBM 제한: 1000만 실행
        )
        
        # 계산된 최적 샷 수 사용
        print(f"\n🎯 Config 기반 실행 계획:")
        print(f"   총 배치 수: {batch_info['num_batches']}")
        print(f"   총 실행 수: {batch_info['total_executions']:,}")
        print(f"   데이터 무결성: {batch_info['data_integrity']}")
        
        # 배치별 실행
        all_results = []
        successful_executions = 0
        failed_executions = 0
        
        for batch_idx in range(batch_info["num_batches"]):
            batch_data = batch_info["batches"][batch_idx]
            batch_circuit_indices = batch_data["circuits"]
            batch_total_shots = batch_data["total_shots"]
            batch_circuit_count = batch_data["circuit_count"]
            
            print(f"\n📦 배치 {batch_idx + 1}/{batch_info['num_batches']} 실행 중...")
            print(f"   회로 수: {batch_circuit_count:,}")
            print(f"   배치 총 실행 수: {batch_total_shots:,}")
            
            # 현재 배치의 회로 선택
            batch_circuits = [qiskit_circuits[i] for i in batch_circuit_indices]
            batch_metadata = [all_circuits[i] for i in batch_circuit_indices]
            
            print(f"   회로 인덱스: {batch_circuit_indices[:5]}{'...' if len(batch_circuit_indices) > 5 else ''}")
            
            # 배치 실행 (각 회로마다 config 기반 샷 수 사용)
            try:
                # 배치 전체를 하나의 작업으로 실행
                print(f"   🚀 배치 {batch_idx + 1} 전체를 하나의 작업으로 실행 중...")
                
                batch_results = run_mega_job(
                    batch_circuits, 
                    batch_metadata, 
                    ibm_backend, 
                    shots=128,  # 기본값 제공 (circuit_shot_requirements가 우선)
                    circuit_shot_requirements=[batch_info["circuit_shot_requirements"][i] for i in batch_circuit_indices]
                )
                
                if batch_results:
                    all_results.extend(batch_results)
                    successful_executions += len(batch_results)
                    print(f"   ✅ 배치 {batch_idx + 1} 성공: {len(batch_results)}개 회로 완료")
                    
                    # 배치 성능 요약
                    batch_fidelities = [r["execution_result"].get("zero_state_probability", 0) for r in batch_results]
                    batch_robust_fidelities = [r["execution_result"].get("robust_fidelity", 0) for r in batch_results]
                    batch_error_rates = [r["execution_result"].get("error_rates", {}).get("total_error_rate", 0) for r in batch_results]
                    
                    print(f"   📊 배치 성능 요약:")
                    print(f"      평균 피델리티: {np.mean(batch_fidelities):.6f}")
                    print(f"      평균 Robust 피델리티: {np.mean(batch_robust_fidelities):.6f}")
                    print(f"      평균 오류율: {np.mean(batch_error_rates):.6f}")
                else:
                    failed_executions += batch_circuit_count
                    print(f"   ❌ 배치 {batch_idx + 1} 실패")
                
                # 배치 결과 저장 (중간 저장으로 데이터 무결성 보장)
                if batch_results:
                    batch_filename = f"batch_{batch_idx + 1}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    save_batch_results(batch_results, batch_filename)
                    print(f"   💾 배치 결과 저장: {batch_filename}")
                    
                    # 매 100개 회로마다 중간 통합 결과 저장
                    if len(all_results) % 100 == 0 and len(all_results) > 0:
                        interim_filename = f"interim_results_{len(all_results)}_circuits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        save_final_results(all_results, interim_filename)
                        print(f"   📊 중간 통합 결과 저장: {interim_filename} ({len(all_results)}개 회로)")
                        
                        # 중간 요약 통계도 저장
                        interim_summary = generate_summary_statistics(all_results)
                        interim_summary_filename = f"interim_summary_{len(all_results)}_circuits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        
                        save_dir = "grid_circuits/mega_results"
                        os.makedirs(save_dir, exist_ok=True)
                        interim_summary_filepath = os.path.join(save_dir, interim_summary_filename)
                        
                        with open(interim_summary_filepath, 'w', encoding='utf-8') as f:
                            json.dump(interim_summary, f, indent=2, default=str, ensure_ascii=False)
                        print(f"   📈 중간 요약 통계 저장: {interim_summary_filepath}")
                
            except Exception as e:
                print(f"   ❌ 배치 {batch_idx + 1} 실행 중 오류: {str(e)}")
                failed_executions += batch_circuit_count
                continue
            
            # 진행 상황 출력
            total_processed = successful_executions + failed_executions
            progress = total_processed / len(qiskit_circuits) * 100
            print(f"   📊 전체 진행률: {progress:.1f}% ({total_processed}/{len(qiskit_circuits)})")
            
            # 실제 사용된 샷 수 계산
            actual_shots_used = sum([batch_info["circuit_shot_requirements"][i] for i in range(successful_executions)])
            print(f"   🎯 누적 샷 사용량: {actual_shots_used:,}")
            
            # 배치 간 대기 시간 (백엔드 부하 분산)
            if batch_idx < batch_info["num_batches"] - 1:
                wait_time = 30  # 30초 대기
                print(f"   ⏳ 다음 배치까지 {wait_time}초 대기...")
                time.sleep(wait_time)
        
        print(f"\n✅ 전체 실행 완료!")
        print(f"   성공한 회로: {successful_executions}/{len(qiskit_circuits)}")
        print(f"   실패한 회로: {failed_executions}")
        print(f"   성공률: {successful_executions/len(qiskit_circuits)*100:.1f}%")
        
        # 실제 사용된 총 샷 수 계산
        actual_total_shots = sum([batch_info["circuit_shot_requirements"][i] for i in range(successful_executions)])
        print(f"   실제 사용된 총 샷 수: {actual_total_shots:,}")
        print(f"   목표 대비 샷 사용률: {(actual_total_shots)/target_total_shots*100:.1f}%")
        
        if not all_results:
            print("⚠️ 실행된 결과가 없습니다.")
            return
        
        # 5️⃣ 결과 분석 및 저장
        print("\n5️⃣ 결과 분석 및 저장")
        
        # 최종 결과 저장
        final_filename = f"mega_job_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_final_results(all_results, final_filename)
        print(f"최종 결과 저장됨: {final_filename}")
        
        # 2큐빗 게이트 비율별 분석
        analyze_two_qubit_ratio_results(all_results)
        
        # 요약 통계 생성 (실제 샷 수 정보 포함)
        summary_stats = generate_summary_statistics(all_results)
        
        # 실제 사용된 총 샷 수 계산
        actual_total_shots_final = sum([batch_info["circuit_shot_requirements"][i] for i in range(successful_executions)])
        
        summary_stats["execution_info"] = {
            "target_total_shots": target_total_shots,
            "actual_total_shots": actual_total_shots_final,
            "shots_efficiency": actual_total_shots_final / target_total_shots,
            "data_integrity": batch_info["data_integrity"],
            "batch_count": batch_info["num_batches"],
            "config_based_shots": True,
            "shots_adequacy": batch_info["shots_adequacy"]
        }
        
        # 요약 통계 저장
        summary_filename = f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 저장 디렉토리 확인 및 생성
        save_dir = "grid_circuits/mega_results"
        os.makedirs(save_dir, exist_ok=True)
        summary_filepath = os.path.join(save_dir, summary_filename)
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, default=str, ensure_ascii=False)
        print(f"✅ 요약 통계 저장됨: {summary_filepath}")
        
        # 실행 시간 계산
        total_execution_time = time.time() - start_time
        
        print(f"\n🎉 Config 기반 메가잡 완료!")
        print(f"   총 실행 시간: {total_execution_time/3600:.1f}시간")
        print(f"   목표 샷 수: {target_total_shots:,}")
        print(f"   실제 사용 샷 수: {actual_total_shots_final:,}")
        print(f"   샷 효율성: {actual_total_shots_final/target_total_shots*100:.1f}%")
        print(f"   성공한 회로: {successful_executions}")
        print(f"   회로당 평균 시간: {total_execution_time/successful_executions:.1f}초")
        print(f"   시간당 처리 회로: {successful_executions/(total_execution_time/3600):.1f}개")
        print(f"   데이터 무결성: {batch_info['data_integrity']}")
        
        # 최적 2큐빗 게이트 비율 추천
        if len(all_results) > 0:
            print("\n📊 2큐빗 게이트 비율별 성능 요약:")
            ratio_performance = {}
            
            for result in all_results:
                ratio = result["circuit_info"].get("two_qubit_ratio", 0)
                if ratio not in ratio_performance:
                    ratio_performance[ratio] = {
                        "fidelities": [],
                        "robust_fidelities": [],
                        "error_rates": [],
                        "expressibilities": []
                    }
                
                exec_result = result["execution_result"]
                ratio_performance[ratio]["fidelities"].append(exec_result.get("zero_state_probability", 0))
                ratio_performance[ratio]["robust_fidelities"].append(exec_result.get("robust_fidelity", 0))
                ratio_performance[ratio]["error_rates"].append(exec_result.get("error_rates", {}).get("total_error_rate", 0))
                
                if exec_result.get("expressibility"):
                    ratio_performance[ratio]["expressibilities"].append(
                        exec_result["expressibility"].get("expressibility_value", 0)
                    )
             
             # 비율별 평균 성능 계산
            for ratio in sorted(ratio_performance.keys()):
                perf = ratio_performance[ratio]
                avg_fidelity = np.mean(perf["fidelities"]) if perf["fidelities"] else 0
                avg_robust_fidelity = np.mean(perf["robust_fidelities"]) if perf["robust_fidelities"] else 0
                avg_error_rate = np.mean(perf["error_rates"]) if perf["error_rates"] else 0
                avg_expressibility = np.mean(perf["expressibilities"]) if perf["expressibilities"] else 0
                
                print(f"   {ratio:.1%} 비율:")
                print(f"     평균 피델리티: {avg_fidelity:.6f}")
                print(f"     평균 Robust 피델리티: {avg_robust_fidelity:.6f}")
                print(f"     평균 오류율: {avg_error_rate:.6f}")
                if avg_expressibility > 0:
                    print(f"     평균 표현력: {avg_expressibility:.6f}")
             
             # 최적 비율 추천
            best_ratio = max(ratio_performance.keys(), 
                           key=lambda r: np.mean(ratio_performance[r]["robust_fidelities"]) if ratio_performance[r]["robust_fidelities"] else 0)
            print(f"\n🏆 추천 2큐빗 게이트 비율: {best_ratio:.1%} (Robust 피델리티 기준)")
            print(f"    Config 기반 적정 샷 수로 측정하여 통계적 신뢰성 확보")
         
        print("\n" + "="*60)
        print("최적화된 800만 샷 메가잡 실험 완료!")
        print(f"데이터 무결성: {batch_info['data_integrity']}")
        print("="*60)
         
    except Exception as e:
        print(f"\n❌ 메가잡 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_two_qubit_ratio_results(all_results):
    """2큐빗 게이트 비율별 결과 분석 (1800개 회로)"""
    print("\n===== 2큐빗 게이트 비율별 성능 분석 =====")
    
    # 설정별 그룹화
    config_groups = {}
    for result in all_results:
        config_group = result["circuit_info"]["config_group"]
        if config_group not in config_groups:
            config_groups[config_group] = []
        config_groups[config_group].append(result)
    
    print(f"총 {len(all_results)}개 회로 결과 분석")
    print(f"설정 그룹 수: {len(config_groups)}")
    
    # 각 설정별 상세 분석
    for config_group in sorted(config_groups.keys()):
        group_results = config_groups[config_group]
        
        if not group_results:
            continue
            
        # 설정 정보 파싱
        parts = config_group.split('_')
        n_qubits = int(parts[0][1:])  # q5 -> 5
        depth = int(parts[1][1:])     # d10 -> 10
        ratio_percent = int(parts[2][1:])  # r30 -> 30
        ratio = ratio_percent / 100.0
        
        print(f"\n📊 {config_group} ({n_qubits}큐빗, 깊이{depth}, 2큐빗비율{ratio:.1%}):")
        print(f"   회로 수: {len(group_results)}개")
        
        # 피델리티 통계
        fidelities = [r["execution_result"]["zero_state_probability"] for r in group_results]
        print(f"   피델리티: {np.mean(fidelities):.6f} ± {np.std(fidelities):.6f}")
        print(f"   피델리티 범위: [{np.min(fidelities):.6f}, {np.max(fidelities):.6f}]")
        
        # 오류율 통계 (있는 경우)
        if "error_rates" in group_results[0]["execution_result"]:
            error_rates = [r["execution_result"]["error_rates"]["total_error_rate"] for r in group_results]
            print(f"   총 오류율: {np.mean(error_rates):.6f} ± {np.std(error_rates):.6f}")
            
            # 비트플립 오류율
            bit_flip_rates = [r["execution_result"]["error_rates"]["bit_flip_error_rate"] for r in group_results]
            print(f"   비트플립 오류율: {np.mean(bit_flip_rates):.6f} ± {np.std(bit_flip_rates):.6f}")
            
            # 위상 오류율
            phase_rates = [r["execution_result"]["error_rates"]["phase_error_rate"] for r in group_results]
            print(f"   위상 오류율: {np.mean(phase_rates):.6f} ± {np.std(phase_rates):.6f}")
        
        # 측정 상태 수
        measured_states = [r["execution_result"]["measured_states"] for r in group_results]
        print(f"   측정 상태 수: {np.mean(measured_states):.0f} ± {np.std(measured_states):.0f}")
    
    # 2큐빗 게이트 비율별 종합 분석
    print(f"\n🔍 2큐빗 게이트 비율별 종합 분석:")
    
    ratio_analysis = {}
    for ratio in [0.1, 0.3, 0.5]:
        ratio_results = []
        for result in all_results:
            if abs(result["circuit_info"]["two_qubit_ratio_target"] - ratio) < 0.01:
                ratio_results.append(result)
        
        if ratio_results:
            fidelities = [r["execution_result"]["zero_state_probability"] for r in ratio_results]
            
            ratio_analysis[ratio] = {
                "count": len(ratio_results),
                "fidelity_mean": np.mean(fidelities),
                "fidelity_std": np.std(fidelities)
            }
            
            if "error_rates" in ratio_results[0]["execution_result"]:
                error_rates = [r["execution_result"]["error_rates"]["total_error_rate"] for r in ratio_results]
                ratio_analysis[ratio]["error_rate_mean"] = np.mean(error_rates)
                ratio_analysis[ratio]["error_rate_std"] = np.std(error_rates)
            
            print(f"   {ratio:.1%} 비율 ({len(ratio_results)}개 회로):")
            print(f"     평균 피델리티: {np.mean(fidelities):.6f} ± {np.std(fidelities):.6f}")
            if "error_rates" in ratio_results[0]["execution_result"]:
                print(f"     평균 오류율: {np.mean(error_rates):.6f} ± {np.std(error_rates):.6f}")
    
    # 최적 설정 추천
    print(f"\n🏆 성능 비교 및 추천:")
    
    # 전체 결과에서 최고 성능 찾기
    best_fidelity_result = max(all_results, key=lambda r: r["execution_result"]["zero_state_probability"])
    print(f"   최고 피델리티: {best_fidelity_result['execution_result']['zero_state_probability']:.6f}")
    print(f"     설정: {best_fidelity_result['circuit_info']['config_group']}")
    
    if "error_rates" in all_results[0]["execution_result"]:
        best_error_result = min(all_results, key=lambda r: r["execution_result"]["error_rates"]["total_error_rate"])
        print(f"   최저 오류율: {best_error_result['execution_result']['error_rates']['total_error_rate']:.6f}")
        print(f"     설정: {best_error_result['circuit_info']['config_group']}")
    
    # 2큐빗 게이트 비율별 추천
    print(f"\n💡 2큐빗 게이트 비율 추천:")
    if ratio_analysis:
        best_ratio_fidelity = max(ratio_analysis.keys(), key=lambda r: ratio_analysis[r]["fidelity_mean"])
        print(f"   피델리티 기준 최적 비율: {best_ratio_fidelity:.1%}")
        
        if "error_rate_mean" in ratio_analysis[0.1]:
            best_ratio_error = min(ratio_analysis.keys(), key=lambda r: ratio_analysis[r]["error_rate_mean"])
            print(f"   오류율 기준 최적 비율: {best_ratio_error:.1%}")
    
    print("="*60)

def save_batch_results(batch_results, filename):
    """배치 결과를 JSON 파일로 저장"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = "grid_circuits/mega_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # 전체 파일 경로
        filepath = os.path.join(save_dir, filename)
        
        # 결과를 JSON 직렬화 가능한 형태로 변환
        serializable_results = []
        for result in batch_results:
            serializable_result = {}
            
            # 회로 정보 처리 (안전한 방식)
            if "circuit_info" in result:
                circuit_info = result["circuit_info"].copy()
                # numpy 배열을 리스트로 변환
                if "params" in circuit_info and hasattr(circuit_info["params"], "__iter__"):
                    try:
                        circuit_info["params"] = [float(p) for p in circuit_info["params"]]
                    except:
                        circuit_info["params"] = []
                serializable_result["circuit_info"] = circuit_info
            
            # 실행 결과 처리 (안전한 방식)
            if "execution_result" in result:
                execution_result = result["execution_result"].copy()
                # 복잡한 객체 제거
                if "result_obj" in execution_result:
                    del execution_result["result_obj"]
                serializable_result["execution_result"] = execution_result
            
            # 배치 정보 (안전한 처리)
            serializable_result["batch_info"] = result.get("batch_info", {
                "batch_id": "unknown",
                "timestamp": datetime.now().isoformat()
            })
            
            # 기타 필드들 안전하게 복사
            for key, value in result.items():
                if key not in ["circuit_info", "execution_result", "batch_info"]:
                    try:
                        # JSON 직렬화 가능한지 테스트
                        json.dumps(value, default=str)
                        serializable_result[key] = value
                    except:
                        # 직렬화 불가능한 경우 문자열로 변환
                        serializable_result[key] = str(value)
            
            serializable_results.append(serializable_result)
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"  ✅ 배치 결과 저장 완료: {filepath} ({len(batch_results)}개 회로)")
        
    except Exception as e:
        print(f"  ⚠️ 배치 결과 저장 오류: {str(e)}")
        # 상세한 오류 정보 출력
        import traceback
        print(f"  상세 오류: {traceback.format_exc()}")
        
        # 최소한의 정보라도 저장 시도
        try:
            minimal_data = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "circuit_count": len(batch_results),
                "basic_info": [{"circuit_id": i, "status": "error"} for i in range(len(batch_results))]
            }
            
            error_filepath = os.path.join(save_dir, f"error_{filename}")
            with open(error_filepath, 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, indent=2, ensure_ascii=False)
            print(f"  📝 오류 정보 저장: {error_filepath}")
        except:
            print(f"  ❌ 오류 정보 저장도 실패")

def save_final_results(all_results, filename):
    """최종 결과를 JSON 파일로 저장"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = "grid_circuits/mega_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # 전체 파일 경로
        filepath = os.path.join(save_dir, filename)
        
        # 결과를 JSON 직렬화 가능한 형태로 변환
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            
            # 회로 정보 처리
            circuit_info = result["circuit_info"].copy()
            # numpy 배열을 리스트로 변환
            if "params" in circuit_info:
                circuit_info["params"] = [float(p) for p in circuit_info["params"]]
            serializable_result["circuit_info"] = circuit_info
            
            # 실행 결과 처리
            execution_result = result["execution_result"].copy()
            # 복잡한 객체 제거
            if "result_obj" in execution_result:
                del execution_result["result_obj"]
            serializable_result["execution_result"] = execution_result
            
            # 배치 정보
            serializable_result["batch_info"] = result.get("batch_info", {})
            
            serializable_results.append(serializable_result)
        
        # 요약 통계 추가
        summary_stats = generate_summary_statistics(all_results)
        
        final_data = {
            "experiment_info": {
                "total_circuits": len(all_results),
                "experiment_type": "two_qubit_ratio_test",
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "n_qubits_list": [5, 7, 10],
                    "depth_list": [5, 10, 15],
                    "two_qubit_ratios": [0.1, 0.3, 0.5],
                    "circuits_per_config": 200
                }
            },
            "summary_statistics": summary_stats,
            "detailed_results": serializable_results
        }
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ 최종 결과 저장 완료: {filepath} ({len(all_results)}개 회로)")
        
    except Exception as e:
        print(f"⚠️ 최종 결과 저장 오류: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_summary_statistics(all_results):
    """
    800만 샷 실험 결과의 요약 통계 생성
    
    Args:
        all_results (list): 모든 실행 결과
        
    Returns:
        dict: 요약 통계
    """
    if not all_results:
        return {"error": "결과가 없습니다."}
    
    print(f"\n📊 {len(all_results)}개 회로 결과의 요약 통계 생성 중...")
    
    # 기본 통계
    summary = {
        "experiment_info": {
            "total_circuits": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "800만 샷 2큐빗 게이트 비율 테스트"
        },
        "circuit_statistics": {},
        "performance_statistics": {},
        "two_qubit_ratio_analysis": {},
        "error_analysis": {},
        "recommendations": {}
    }
    
    # 회로 통계
    n_qubits_list = [r["circuit_info"]["n_qubits"] for r in all_results]
    depths = [r["circuit_info"]["depth"] for r in all_results]
    two_qubit_ratios = [r["circuit_info"].get("two_qubit_ratio", 0) for r in all_results]
    
    summary["circuit_statistics"] = {
        "qubit_range": {"min": min(n_qubits_list), "max": max(n_qubits_list)},
        "depth_range": {"min": min(depths), "max": max(depths)},
        "two_qubit_ratio_range": {"min": min(two_qubit_ratios), "max": max(two_qubit_ratios)},
        "unique_configurations": len(set((q, d, r) for q, d, r in zip(n_qubits_list, depths, two_qubit_ratios)))
    }
    
    # 성능 통계
    fidelities = []
    robust_fidelities = []
    total_error_rates = []
    expressibilities = []
    execution_times = []
    
    for result in all_results:
        exec_result = result["execution_result"]
        
        fidelities.append(exec_result.get("zero_state_probability", 0))
        robust_fidelities.append(exec_result.get("robust_fidelity", 0))
        
        error_rates = exec_result.get("error_rates", {})
        total_error_rates.append(error_rates.get("total_error_rate", 0))
        
        if exec_result.get("expressibility"):
            expressibilities.append(exec_result["expressibility"].get("expressibility_value", 0))
        
        execution_times.append(result.get("execution_time", 0))
    
    summary["performance_statistics"] = {
        "fidelity": {
            "mean": float(np.mean(fidelities)),
            "std": float(np.std(fidelities)),
            "min": float(np.min(fidelities)),
            "max": float(np.max(fidelities)),
            "median": float(np.median(fidelities))
        },
        "robust_fidelity": {
            "mean": float(np.mean(robust_fidelities)),
            "std": float(np.std(robust_fidelities)),
            "min": float(np.min(robust_fidelities)),
            "max": float(np.max(robust_fidelities)),
            "median": float(np.median(robust_fidelities))
        },
        "total_error_rate": {
            "mean": float(np.mean(total_error_rates)),
            "std": float(np.std(total_error_rates)),
            "min": float(np.min(total_error_rates)),
            "max": float(np.max(total_error_rates)),
            "median": float(np.median(total_error_rates))
        }
    }
    
    if expressibilities:
        summary["performance_statistics"]["expressibility"] = {
            "mean": float(np.mean(expressibilities)),
            "std": float(np.std(expressibilities)),
            "min": float(np.min(expressibilities)),
            "max": float(np.max(expressibilities)),
            "median": float(np.median(expressibilities))
        }
    
    # 2큐빗 게이트 비율별 분석
    ratio_groups = {}
    for result in all_results:
        ratio = result["circuit_info"].get("two_qubit_ratio", 0)
        if ratio not in ratio_groups:
            ratio_groups[ratio] = {
                "fidelities": [],
                "robust_fidelities": [],
                "error_rates": [],
                "expressibilities": [],
                "count": 0
            }
        
        exec_result = result["execution_result"]
        ratio_groups[ratio]["fidelities"].append(exec_result.get("zero_state_probability", 0))
        ratio_groups[ratio]["robust_fidelities"].append(exec_result.get("robust_fidelity", 0))
        ratio_groups[ratio]["error_rates"].append(exec_result.get("error_rates", {}).get("total_error_rate", 0))
        
        if exec_result.get("expressibility"):
            ratio_groups[ratio]["expressibilities"].append(exec_result["expressibility"].get("expressibility_value", 0))
        
        ratio_groups[ratio]["count"] += 1
    
    # 비율별 통계 계산
    for ratio, data in ratio_groups.items():
        summary["two_qubit_ratio_analysis"][f"{ratio:.1%}"] = {
            "count": data["count"],
            "fidelity_mean": float(np.mean(data["fidelities"])),
            "robust_fidelity_mean": float(np.mean(data["robust_fidelities"])),
            "error_rate_mean": float(np.mean(data["error_rates"])),
            "fidelity_std": float(np.std(data["fidelities"])),
            "robust_fidelity_std": float(np.std(data["robust_fidelities"])),
            "error_rate_std": float(np.std(data["error_rates"]))
        }
        
        if data["expressibilities"]:
            summary["two_qubit_ratio_analysis"][f"{ratio:.1%}"]["expressibility_mean"] = float(np.mean(data["expressibilities"]))
            summary["two_qubit_ratio_analysis"][f"{ratio:.1%}"]["expressibility_std"] = float(np.std(data["expressibilities"]))
    
    # 오류 분석
    high_error_circuits = [r for r in all_results if r["execution_result"].get("error_rates", {}).get("total_error_rate", 0) > 0.1]
    low_fidelity_circuits = [r for r in all_results if r["execution_result"].get("zero_state_probability", 0) < 0.5]
    
    summary["error_analysis"] = {
        "high_error_circuits_count": len(high_error_circuits),
        "high_error_circuits_percentage": len(high_error_circuits) / len(all_results) * 100,
        "low_fidelity_circuits_count": len(low_fidelity_circuits),
        "low_fidelity_circuits_percentage": len(low_fidelity_circuits) / len(all_results) * 100
    }
    
    # 추천사항
    best_ratio_by_fidelity = max(ratio_groups.keys(), key=lambda r: np.mean(ratio_groups[r]["fidelities"]))
    best_ratio_by_robust_fidelity = max(ratio_groups.keys(), key=lambda r: np.mean(ratio_groups[r]["robust_fidelities"]))
    best_ratio_by_low_error = min(ratio_groups.keys(), key=lambda r: np.mean(ratio_groups[r]["error_rates"]))
    
    summary["recommendations"] = {
        "best_ratio_for_fidelity": f"{best_ratio_by_fidelity:.1%}",
        "best_ratio_for_robust_fidelity": f"{best_ratio_by_robust_fidelity:.1%}",
        "best_ratio_for_low_error": f"{best_ratio_by_low_error:.1%}",
        "overall_recommendation": f"{best_ratio_by_robust_fidelity:.1%}",
        "recommendation_reason": "Robust 피델리티가 노이즈 환경에서 더 신뢰할 수 있는 지표입니다."
    }
    
    print(f"✅ 요약 통계 생성 완료")
    print(f"   분석된 회로 수: {len(all_results)}")
    print(f"   고유 설정 수: {summary['circuit_statistics']['unique_configurations']}")
    print(f"   평균 피델리티: {summary['performance_statistics']['fidelity']['mean']:.6f}")
    print(f"   평균 Robust 피델리티: {summary['performance_statistics']['robust_fidelity']['mean']:.6f}")
    print(f"   추천 2큐빗 게이트 비율: {summary['recommendations']['overall_recommendation']}")
    
    return summary

if __name__ == "__main__":
    print("🚀 Mega Job Generator 시작!")
    print("📁 디렉토리 설정 중...")
    setup_directories()
    print("\n🎯 메가잡 실행 시작!")
    run_mega_job_generator() 