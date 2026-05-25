#include "interlock_controller.hpp"
#include <iostream>
#include <algorithm>

namespace aurore {

InterlockController::InterlockController()
    : current_state_(InterlockState::UNKNOWN), 
      last_change_timestamp_ns_(0), 
      internal_watchdog_timeout_ms_(1000.0), // Default 1 second timeout
      is_initialized_(false) {}

InterlockController::~InterlockController() {
    // Cleanup resources (e.g., closing GPIO file handles in a real implementation)
}

bool InterlockController::init() {
    // Simulated hardware initialization check: e.g., checking if required GPIO pins are available.
    if (/* Hardware setup fails */ false) {
        std::cerr << "FATAL: Could not initialize Interlock Controller hardware access.\n";
        return false;
    }

    // Initialize internal state and set initial safe values
    current_state_ = InterlockState::UNKNOWN;
    last_change_timestamp_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    is_initialized_ = true;
    std::cout << "InterlockController initialized successfully. Awaiting safety circuit confirmation." << std::endl;
    return true;
}

bool InterlockController::watchdog_feed(float elapsed_ms) {
    if (!is_initialized_) return false;

    // In a real system, this function would write a "keep-alive" signal to the hardware.
    // Here, we just update internal timing metrics for simulation purposes.
    // The check is that enough time has passed since last change (implicit).
    return true; 
}

InterlockStatus InterlockController::get_status() const {
    InterlockStatus status;
    status.state = current_state_;
    status.last_change_ns = last_change_timestamp_ns_;
    // Check if the state is OPEN (the safe, inhibited default)
    status.actuation_inhibited = (current_state_ == InterlockState::OPEN || 
                                 current_state_ == InterlockState::FAULT);
    return status;
}

bool InterlockController::force_state(InterlockState target_state) {
    if (!is_initialized_) return false;

    // Only allow force state changes if the current state is already unsafe (e.g., for testing fault recovery)
    if (current_state_ == InterlockState::CLOSED && target_state != InterlockState::OPEN) {
        std::cout << "WARNING: Attempted to force state while physically closed." << std::endl;
        return false;
    }

    // Update internal state and timestamp
    current_state_ = target_state;
    last_change_timestamp_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    return true;
}

InterlockState InterlockController::read_physical_input() const {
    // Placeholder: In a real implementation, this reads from GPIO 17 (INTERLOCK_IN)
    // Logic: If signal is LOW (active-low), the circuit is closed.
    // We simulate that the default state upon boot is OPEN for safety.
    return current_state_; // Currently returns internal state as simulation placeholder
}

void InterlockController::run_monitoring_cycle(float elapsed_ms) {
    if (!is_initialized_) return;

    InterlockState new_physical_state = read_physical_input();
    InterlockState old_state = current_state_;

    // 1. Check for state change (The core detection logic)
    if (new_physical_state != old_state && old_state != InterlockState::UNKNOWN) {
        std::cout << "[INTERLOCK] State Transition Detected: " 
                  << static_cast<int>(old_state) << " -> " 
                  << static_cast<int>(new_physical_state) << std::endl;
        current_state_ = new_physical_state;
        last_change_timestamp_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }

    // 2. Check for watchdog timeout (Simulated)
    if (elapsed_ms > internal_watchdog_timeout_ms_) {
        current_state_ = InterlockState::FAULT; // Force to FAULT on timeout
        std::cerr << "[INTERLOCK] *** WATCHDOG TIMEOUT DETECTED. FORCING TO FAULT STATE! ***" << std::endl;
    }
}

} // namespace aurore