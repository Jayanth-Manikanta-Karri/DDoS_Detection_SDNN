import random
import numpy as np
import tensorflow as tf
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, ether_types

# Define the DNN model path
MODEL_PATH = 'Model/cnnModel.h5'

class DDoSDetectionController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSDetectionController, self).__init__(*args, **kwargs)
        # Load the saved DNN model
        self.dnn_model = tf.keras.models.load_model(MODEL_PATH)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        # Extract relevant features from the packet
        src_ip = pkt.get_protocol(ipv4.ipv4).src
        dst_ip = pkt.get_protocol(ipv4.ipv4).dst
        src_port = pkt.get_protocol(tcp.tcp).src_port
        dst_port = pkt.get_protocol(tcp.tcp).dst_port

        # Preprocess the features
        features = np.array([src_ip, dst_ip, src_port, dst_port])
        preprocessed_features = self.preprocess_features(features)

        # Pass the preprocessed features to the DNN model for prediction
        prediction = self.dnn_model.predict(preprocessed_features)

        # Check if the prediction indicates a DDoS attack
        if prediction > 0.5:
            # Block the incoming packet from the source
            self.logger.info("DDoS attack detected: Packet from %s to %s", src_ip, dst_ip)
            self.block_source_packet(datapath, msg, src_ip)

    def preprocess_features(self, features):
        # Extract features from the raw feature vector
        src_ip, dst_ip, src_port, dst_port = features

        # Normalize numerical features
        src_port_normalized = self.normalize_port(src_port)
        dst_port_normalized = self.normalize_port(dst_port)

        # Convert IP addresses to numerical values
        src_ip_numeric = self.ip_to_numeric(src_ip)
        dst_ip_numeric = self.ip_to_numeric(dst_ip)

        # Combine all preprocessed features into a single array
        preprocessed_features = np.array([src_ip_numeric, dst_ip_numeric, src_port_normalized, dst_port_normalized])
        return preprocessed_features

    def normalize_port(self, port):
        # Normalize port to a range between 0 and 1
        return port / 65535.0  # Assuming port ranges from 0 to 65535

    def ip_to_numeric(self, ip_address):
        # Convert IP address to a numerical value
        ip_parts = [int(part) for part in ip_address.split('.')]
        return sum(ip_parts[i] << (24 - 8 * i) for i in range(4))

    def block_source_packet(self, datapath, msg, src_ip):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Blocking incoming packets from the source IP
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        actions = []

        # Drop the packet
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_CLEAR_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, table_id=0, priority=10, match=match, instructions=inst)
        datapath.send_msg(mod)

        # Log the information
        self.logger.info("Blocked incoming packets from source IP: %s", src_ip)