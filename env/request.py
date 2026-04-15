from env.network import Node, Link
from env.vnf import VNF
import config 

class Request:
    def __init__(self, name: str, arrival_time: float, delay_max: float, start_node: str, end_node: str, VNFs: list[VNF], bandwidth: float = 0):
        '''
        Thiết lập request
        :param name: Id
        :param arrival_time: Thời gian đến
        :param delay_max: Thời gian trễ tối đa
        :param start_node: Node bắt đầu
        :param end_node: Node kết thúc
        :param VNFs: List VNFs
        :param bandwidth: Băng thông tối đa cần thiết
        '''
        self.name = name
        self.arrival_time = arrival_time
        self.delay_max = delay_max
        self.end_time = self.arrival_time + self.delay_max

        self.start_node = start_node
        self.end_node = end_node
        self.vnfs = VNFs
        self.bw = bandwidth

class SFC:
    def __init__(self, request: Request):
        '''
        Khởi tạo SFC request thực hiện trên mạng
        '''
        self.request = request
        self.route_nodes: list[Node] = []
        self.route_links: list[Link] = []
        self.vnf_indices: list[int] = []

    def deploy(self, T: float):
        '''
        Triển khai request trên mạng tại thời điểm T
        '''
        pass

    def undeploy(self, T: float):
        pass
    
    def __check_capacity_constraints(self, T):
        for node in self.route_nodes:
            if node.check_violated(T) is True:
                return False
        for link in self.route_links:
            if link.check_violated(T) is True:
                return False
        return True
    
    def check_feasibility(self, T):
        '''
        Kiểm tra request thực hiện tại timeslot T có thỏa mãn không.
        :param T: timeslot
        '''
        self.deploy(T)
        valid = self.__check_capacity_constraints(T)
        self.undeploy(T)
        return valid

    def to_dict(self):
        return {
            'nodes': [node.name for node in self.route_nodes],
            'vnfs': self.vnf_indices
        }
    
class ListOfRequests:
    def __init__(self):
        self.requests = []

    def add_request(self, request: Request):
        self.requests.append(request)

    def _get_arrival_time(self, n: Request):
        return n.arrival_time

    def sort_requests(self):
        self.requests.sort(key = self._get_arrival_time)