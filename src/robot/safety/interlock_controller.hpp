#ifndef INTERLOCK_CONTROLLER_HPP
#define INTERLOCK_CONTROLLER_HPP

#include <cstdint>
#include <chrono>
#include <optional>

namespace aurore {

/**
 * @brief Enumeration of possible safety states, mirroring the physical hardware.
 */
enum class InterlockState : uint8_t {
    UNKNOWN,   // Initial state or unknown connection
    OPEN,       // Safety circuit is open - Actuation MUST be inhibited (Safe State)
    CLOSED,     // Safety circuit is closed - Actuation MAY proceed
    FAULT        // Hardware fault detected (e.g., watchdog failure, unexpected input)
};

/**
 * @brief Status structure mirroring the physical hardware readouts.
 */
struct InterlockStatus {
    InterlockState state;           ///< Current logical safety state.
    uint64_t last_change_ns;        ///< Timestamp of last reported state change (nanoseconds).
    bool actuation_inhibited;       ///< True if the system cannot command motion/action.
};

/**
 * @brief Abstract Interlock Controller class for simulating physical safety constraints.
 * 
 * This class enforces that no action can be taken unless a simulated 'safety circuit' is closed.
 */
class InterlockController {
public:
    InterlockController();
    ~InterlockController();

    /**
     * @brief Initializes the controller, simulating hardware connection checks.
     * @return true if initialized successfully, false otherwise (e.g., missing GPIO).
     */
    bool init();

    /**
     * @brief Runs a single monitoring cycle (simulating 1kHz thread tick). Must be called regularly.
     * @param elapsed_ms Time elapsed since the last call (for simulating real-time drift).
     * @return true if monitoring was successful, false if hardware fault occurred.
     */
    bool watchdog_feed(float elapsed_ms);

    /**
     * @brief Checks if current simulated state allows for physical action.
     * @return The InterlockStatus reflecting the current safety condition.
     */
    InterlockStatus get_status() const;

    /**
     * @brief Attempts to force a specific internal state (for testing/simulation logic).
     * @param target_state The desired state.
     * @return true if force was successful.
     */
    bool force_state(InterlockState target_state);


private:
    // State variables mimicking hardware registers
    volatile InterlockState current_state_;
    uint64_t last_change_timestamp_ns_;
    float internal_watchdog_timeout_ms_;
    
    // Simulation state
    bool is_initialized_;

    /**
     * @brief Simulates reading the physical safety circuit input. 
     * In a real port, this would read from an ADC or GPIO pin.
     */
    InterlockState read_physical_input() const;

    /**
     * @brief Updates internal state based on simulated hardware readings and time checks.
     */
    void run_monitoring_cycle(float elapsed_ms);
};

} // namespace aurore

#endif // INTERLOCK_CONTROLLER_HPP