//! A bunch of statily defined metrics. Defined here because they are used in
//! both macros and handwritten code.

use hyperactor_telemetry::declare_static_counter;

declare_static_counter!(MESSAGES_SENT, "messages_sent");
declare_static_counter!(MESSAGES_RECEIVED, "messages_received");
declare_static_counter!(MESSAGE_HANDLE_ERRORS, "message_handle_errors");
declare_static_counter!(MESSAGE_RECEIVE_ERRORS, "message_receive_errors");
